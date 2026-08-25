"""Reusable collection and aligned vector input validation."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .helpers import read_json


def validate_collection(info: Dict[str, Any], expected_size: int, expected_distance: str) -> None:
    vectors = info.get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
    actual_size = vectors.get("size")
    actual_distance = str(vectors.get("distance", ""))
    if int(actual_size or -1) != expected_size:
        raise ValueError("Qdrant vector size uyuşmuyor: %r" % actual_size)
    if actual_distance.casefold() != expected_distance.casefold():
        raise ValueError("Qdrant distance uyuşmuyor: %r" % actual_distance)


def validate_vector_inputs(
    articles: List[Dict[str, Any]], assignments: List[Dict[str, str]],
    embeddings: np.ndarray, expected_count: int, *, quality_path: Path,
    metadata_path: Path, vector_label: str = "Embedding", require_nonempty_id: bool = True,
) -> str:
    """Validate 50K row alignment, fixed 768D vectors and dataset provenance."""
    if len(articles) != expected_count:
        raise ValueError("Makale sayısı yanlış: %d" % len(articles))
    if len(assignments) != expected_count:
        raise ValueError("Assignment sayısı yanlış: %d" % len(assignments))
    if embeddings.shape != (expected_count, 768):
        raise ValueError("%s şekli yanlış: %r" % (vector_label, embeddings.shape))
    if embeddings.dtype != np.float32:
        raise ValueError("%s dtype float32 değil." % vector_label)
    if not np.isfinite(embeddings).all():
        raise ValueError("%s NaN/Inf içeriyor." % vector_label)

    dataset_hash = str(read_json(quality_path).get("dataset_sha256", ""))
    embedding_hash = str(read_json(metadata_path).get("dataset_sha256", ""))
    if not dataset_hash:
        raise ValueError("Dataset SHA-256 bulunamadı.")
    if dataset_hash != embedding_hash:
        raise ValueError("Dataset ve embedding SHA-256 eşleşmiyor.\nDataset  : %s\nEmbedding: %s" % (dataset_hash, embedding_hash))

    seen_ids = set()
    for row_index, (article, assignment) in enumerate(zip(articles, assignments)):
        article_id = str(article.get("article_id", "")).strip()
        assignment_id = str(assignment.get("article_id", "")).strip()
        assignment_index = int(assignment.get("row_index", "-1"))
        if assignment_index != row_index:
            raise ValueError("Assignment row_index hizası bozuk: beklenen=%d bulunan=%d" % (row_index, assignment_index))
        if article_id != assignment_id:
            raise ValueError("Article ID hizası bozuk: satır=%d jsonl=%s csv=%s" % (row_index, article_id, assignment_id))
        if require_nonempty_id and not article_id:
            raise ValueError("Boş article_id: %d" % row_index)
        if article_id in seen_ids:
            raise ValueError("Tekrarlı article_id: %s" % article_id)
        seen_ids.add(article_id)
    return dataset_hash
