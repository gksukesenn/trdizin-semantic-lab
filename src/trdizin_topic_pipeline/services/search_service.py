"""Web search use-case orchestration; algorithms live in the search package."""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..search.filters import build_filter
from ..search.formatting import format_hybrid_results, format_semantic_points
from ..search.hybrid_search import multi_source_rrf
from ..search.qdrant_store import QdrantRestStore
from ..search.query_encoder import QueryEncoder

ROOT = Path(__file__).resolve().parents[3]
SUMMARY_PATH = ROOT / "outputs" / "final_50k" / "clustering" / "final_topic_pipeline_summary.json"
QUALITY_PATH = ROOT / "outputs" / "final_50k" / "reports" / "dataset_quality_summary.json"


def _optional_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class SearchService:
    """Coordinates encoding, filtering, retrieval, fusion and formatting."""

    def __init__(
        self, config: Dict[str, Any], allow_cpu: bool,
        encoder: Optional[QueryEncoder] = None, store: Optional[QdrantRestStore] = None,
    ) -> None:
        self.config = config
        qdrant = config.get("qdrant", {})
        self.qdrant_url = str(qdrant.get("url", "http://127.0.0.1:6335"))
        self.abstract_collection = str(qdrant.get("collection_name", "trdizin_articles_50000"))
        self.title_collection = str(qdrant.get("title_collection_name", "trdizin_titles_50000"))
        self.bm25_collection = "trdizin_bm25_50000"
        model_id = str(config.get("embedding", {}).get(
            "model_id", "trmteb/turkish-embedding-model-fine-tuned"
        ))

        print("=" * 80)
        print("TR DİZİN SEMANTIC EXPLORER")
        print("=" * 80)
        print("\nQdrant URL       :", self.qdrant_url)
        print("Abstract         :", self.abstract_collection)
        print("Title            :", self.title_collection)
        print("BM25             :", self.bm25_collection)

        self.encoder = encoder or QueryEncoder(model_id, allow_cpu)
        self.device = self.encoder.device
        self.model = self.encoder.model  # Backward-compatible debug/introspection attribute.
        print("Device           :", self.device)
        if self.device == "cuda":
            print("GPU              :", self.encoder.torch.cuda.get_device_name(0))
        print("\nTR-MTEB yükleniyor...")
        print("Model            :", model_id)
        print("model.device     :", self.model.device)
        print("İlk parametre    :", next(self.model.parameters()).device)

        self.store = store or QdrantRestStore(
            base_url=self.qdrant_url,
            timeout_seconds=int(qdrant.get("timeout_seconds", 180)),
        )
        self.quality = _optional_json(QUALITY_PATH)
        self.pipeline_summary = _optional_json(SUMMARY_PATH)
        print("\nDemo backend hazır.")

    def search(
        self, query: str, mode: str, limit: int,
        year_from: Optional[int], year_to: Optional[int], database: Optional[str],
    ) -> Dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("Sorgu boş olamaz.")
        if limit < 1 or limit > 30:
            raise ValueError("Limit 1–30 arasında olmalıdır.")
        if mode not in {"semantic", "hybrid"}:
            raise ValueError("Geçersiz search mode.")

        query_vector, embedding_seconds = self.encoder.encode(query)
        query_filter = build_filter(year_from, year_to, database)
        started = time.perf_counter()

        if mode == "semantic":
            points = self.store.query_points(
                self.abstract_collection, query_vector.tolist(), limit, query_filter
            )
            results = format_semantic_points(points)
        else:
            candidate_limit = max(50, limit * 5)
            ranked_sources = {
                "abstract": self.store.query_points(
                    self.abstract_collection, query_vector.tolist(), candidate_limit, query_filter
                ),
                "title": self.store.query_points(
                    self.title_collection, query_vector.tolist(), candidate_limit, query_filter
                ),
                "bm25": self.store.query_bm25_points(
                    self.bm25_collection, query, candidate_limit, query_filter
                ),
            }
            fused = multi_source_rrf(ranked_sources, rrf_k=60, limit=limit)
            ids = [row.get("qdrant_id") for row in fused if row.get("qdrant_id") is not None]
            full_points = self.store.retrieve_points(
                self.abstract_collection, ids, with_payload=True, with_vector=False
            )
            payloads = {str(point.get("id")): point.get("payload", {}) for point in full_points}
            results = format_hybrid_results(fused, payloads)

        return {
            "query": query, "mode": mode, "result_count": len(results),
            "embedding_seconds": embedding_seconds,
            "search_seconds": time.perf_counter() - started,
            "filter": query_filter, "results": results,
        }

    def status(self) -> Dict[str, Any]:
        abstract_count = self.store.exact_count(self.abstract_collection)
        title_count = self.store.exact_count(self.title_collection)
        bm25_count = self.store.exact_count(self.bm25_collection)
        return {
            "status": "ok", "device": self.device,
            "gpu": self.encoder.torch.cuda.get_device_name(0) if self.device == "cuda" else None,
            "article_count": abstract_count, "abstract_collection_count": abstract_count,
            "title_collection_count": title_count, "bm25_collection_count": bm25_count,
            "vector_size": 768, "dataset_sha256": self.quality.get("dataset_sha256"),
        }
