#!/usr/bin/env python3
"""Abstract ve başlık Qdrant aramalarını RRF ile birleştirir."""

import argparse
import json
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[3]

from trdizin_topic_pipeline.search.hybrid_search import (
    reciprocal_rank_fusion,
)
from trdizin_topic_pipeline.search.qdrant_store import (
    QdrantRestStore,
)
from trdizin_topic_pipeline.search.filters import build_filter
from trdizin_topic_pipeline.search.cli_support import (
    encode_cli_query as encode_query, load_cli_model as load_model, read_json,
    resolve_path as _resolve_path, score_or_dash as display_score,
    select_cli_device as select_device, shorten,
    value_or_dash as display_rank,
)

resolve_path = partial(_resolve_path, ROOT)


DEFAULT_MODEL_ID = (
    "trmteb/turkish-embedding-model-fine-tuned"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Abstract ve title semantic-search sonuçlarını "
            "Reciprocal Rank Fusion ile birleştirir."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/final_50k.json",
    )

    parser.add_argument(
        "--query",
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Gösterilecek nihai sonuç sayısı.",
    )

    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=50,
        help=(
            "Her Qdrant collectionından alınacak aday sayısı."
        ),
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
    )

    parser.add_argument(
        "--year-from",
        type=int,
    )

    parser.add_argument(
        "--year-to",
        type=int,
    )

    parser.add_argument(
        "--database",
    )

    parser.add_argument(
        "--show-abstract",
        action="store_true",
    )

    parser.add_argument(
        "--allow-cpu",
        action="store_true",
    )

    return parser.parse_args()


def print_results(
    query: str,
    results: List[Dict[str, Any]],
    show_abstract: bool,
) -> None:
    print("\n" + "=" * 90)
    print("ABSTRACT + TITLE RRF SONUÇLARI")
    print("=" * 90)

    print("\nSorgu :", query)
    print("Sonuç :", len(results))

    for final_rank, row in enumerate(
        results,
        start=1,
    ):
        payload = row.get(
            "payload",
            {},
        )

        print("\n" + "-" * 90)

        print(
            "%d. Makale ID=%s | RRF=%.6f"
            % (
                final_rank,
                row.get("article_id"),
                float(row.get("rrf_score", 0.0)),
            )
        )

        print(
            "Abstract rank/score : %s / %s"
            % (
                display_rank(
                    row.get("abstract_rank")
                ),
                display_score(
                    row.get("abstract_score")
                ),
            )
        )

        print(
            "Title rank/score    : %s / %s"
            % (
                display_rank(
                    row.get("title_rank")
                ),
                display_score(
                    row.get("title_score")
                ),
            )
        )

        print(
            "Başlık              :",
            payload.get("title_tr"),
        )

        print(
            "Yıl                 :",
            payload.get(
                "publication_year",
                "Bilinmiyor",
            ),
        )

        print(
            "Database            :",
            ", ".join(
                payload.get(
                    "databases",
                    [],
                )
            ),
        )

        print(
            "Atama yöntemi       :",
            payload.get(
                "assignment_method",
            ),
        )

        print(
            "Birincil konu       :",
            payload.get(
                "primary_topic",
            ),
        )

        print(
            "İkincil konu        :",
            payload.get(
                "secondary_topic",
            ),
        )

        if show_abstract:
            abstract = payload.get(
                "abstract_tr",
                "",
            )

            if abstract:
                print(
                    "Abstract            :",
                    shorten(
                        abstract,
                        600,
                    ),
                )
            else:
                print(
                    "Abstract            :",
                    "Başlık adayından geldiği için payload içinde yok.",
                )

    print(
        "\nNot: RRF skoru, semantic skorlar ve konu marjı "
        "kalibre edilmiş olasılık değildir."
    )


def main() -> None:
    args = parse_arguments()

    query = args.query.strip()

    if not query:
        raise ValueError("Sorgu boş olamaz.")

    if args.limit < 1 or args.limit > 100:
        raise ValueError(
            "--limit 1–100 arasında olmalıdır."
        )

    if args.candidate_limit < args.limit:
        raise ValueError(
            "--candidate-limit, --limit değerinden küçük olamaz."
        )

    if args.rrf_k < 1:
        raise ValueError(
            "--rrf-k en az 1 olmalıdır."
        )

    config = read_json(
        resolve_path(args.config)
    )

    qdrant_config = config.get(
        "qdrant",
        {},
    )

    qdrant_url = str(
        qdrant_config.get(
            "url",
            "http://127.0.0.1:6335",
        )
    )

    abstract_collection = str(
        qdrant_config.get(
            "collection_name",
            "trdizin_articles_50000",
        )
    )

    title_collection = str(
        qdrant_config.get(
            "title_collection_name",
            "trdizin_titles_50000",
        )
    )

    model_id = str(
        config.get(
            "embedding",
            {},
        ).get(
            "model_id",
            DEFAULT_MODEL_ID,
        )
    )

    device = select_device(
        args.allow_cpu
    )

    print("=" * 90)
    print("ABSTRACT + TITLE HYBRID SEMANTIC SEARCH")
    print("=" * 90)

    print("\nPython executable         :", sys.executable)
    print("torch version             :", torch.__version__)
    print("torch CUDA build          :", torch.version.cuda)
    print(
        "torch.cuda.is_available() :",
        torch.cuda.is_available(),
    )
    print("Seçilen cihaz             :", device)

    if device == "cuda":
        print(
            "GPU                       :",
            torch.cuda.get_device_name(0),
        )

    model = load_model(
        model_id=model_id,
        device=device,
    )

    query_vector = encode_query(
        model=model,
        query=query,
        device=device,
    )

    query_filter = build_filter(
        year_from=args.year_from,
        year_to=args.year_to,
        database=args.database,
    )

    print("\nQdrant URL                :", qdrant_url)
    print("Abstract collection       :", abstract_collection)
    print("Title collection          :", title_collection)
    print("Aday limiti               :", args.candidate_limit)
    print("Nihai limit               :", args.limit)
    print("RRF k                     :", args.rrf_k)
    print(
        "Filtre                    :",
        (
            json.dumps(
                query_filter,
                ensure_ascii=False,
            )
            if query_filter
            else "yok"
        ),
    )

    store = QdrantRestStore(
        base_url=qdrant_url,
        timeout_seconds=int(
            qdrant_config.get(
                "timeout_seconds",
                180,
            )
        ),
    )

    try:
        abstract_started = time.perf_counter()

        abstract_points = store.query_points(
            collection_name=abstract_collection,
            query_vector=query_vector.tolist(),
            limit=args.candidate_limit,
            query_filter=query_filter,
        )

        abstract_seconds = (
            time.perf_counter()
            - abstract_started
        )

        title_started = time.perf_counter()

        title_points = store.query_points(
            collection_name=title_collection,
            query_vector=query_vector.tolist(),
            limit=args.candidate_limit,
            query_filter=query_filter,
        )

        title_seconds = (
            time.perf_counter()
            - title_started
        )

    finally:
        store.close()

    fusion_started = time.perf_counter()

    results = reciprocal_rank_fusion(
        abstract_points=abstract_points,
        title_points=title_points,
        rrf_k=args.rrf_k,
        limit=args.limit,
    )

    fusion_seconds = (
        time.perf_counter()
        - fusion_started
    )

    print(
        "Abstract Qdrant süresi    : %.4f sn"
        % abstract_seconds
    )

    print(
        "Title Qdrant süresi       : %.4f sn"
        % title_seconds
    )

    print(
        "RRF birleştirme süresi    : %.6f sn"
        % fusion_seconds
    )

    print_results(
        query=query,
        results=results,
        show_abstract=args.show_abstract,
    )


if __name__ == "__main__":
    main()
