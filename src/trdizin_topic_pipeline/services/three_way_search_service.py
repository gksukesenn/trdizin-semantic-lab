#!/usr/bin/env python3
"""Abstract dense + title dense + BM25 sparse aramasını RRF ile birleştirir."""

import argparse
import json
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
    multi_source_rrf,
)

from trdizin_topic_pipeline.search.qdrant_store import (
    QdrantRestStore,
)
from trdizin_topic_pipeline.search.filters import build_filter
from trdizin_topic_pipeline.search.cli_support import (
    encode_cli_query as _encode_query, load_cli_model as load_model, read_json,
    resolve_path as _resolve_path, score_or_dash,
    select_cli_device as select_device, value_or_dash,
)

resolve_path = partial(_resolve_path, ROOT)
encode_query = partial(_encode_query, style="three_way")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/final_50k.json",
    )

    parser.add_argument(
        "--query",
        required=True,
    )

    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
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
        "--allow-cpu",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    query = args.query.strip()

    if not query:
        raise ValueError(
            "Sorgu boş olamaz."
        )

    config = read_json(
        resolve_path(
            args.config
        )
    )

    qdrant = config.get(
        "qdrant",
        {},
    )

    qdrant_url = str(
        qdrant.get(
            "url",
            "http://127.0.0.1:6335",
        )
    )

    abstract_collection = str(
        qdrant.get(
            "collection_name",
            "trdizin_articles_50000",
        )
    )

    title_collection = str(
        qdrant.get(
            "title_collection_name",
            "trdizin_titles_50000",
        )
    )

    bm25_collection = (
        "trdizin_bm25_50000"
    )

    model_id = str(
        config.get(
            "embedding",
            {},
        ).get(
            "model_id",
            "trmteb/turkish-embedding-model-fine-tuned",
        )
    )

    device = select_device(
        args.allow_cpu
    )

    print("=" * 90)
    print(
        "ABSTRACT + TITLE + BM25 HYBRID SEARCH"
    )
    print("=" * 90)

    print("\nCihaz              :", device)

    if device == "cuda":
        print(
            "GPU                :",
            torch.cuda.get_device_name(0),
        )

    model = load_model(model_id, device, style="three_way")

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

    store = QdrantRestStore(
        base_url=qdrant_url,
        timeout_seconds=int(
            qdrant.get(
                "timeout_seconds",
                180,
            )
        ),
    )

    try:
        started = time.perf_counter()

        abstract_points = store.query_points(
            collection_name=abstract_collection,
            query_vector=query_vector.tolist(),
            limit=args.candidate_limit,
            query_filter=query_filter,
        )

        abstract_seconds = (
            time.perf_counter()
            - started
        )

        started = time.perf_counter()

        title_points = store.query_points(
            collection_name=title_collection,
            query_vector=query_vector.tolist(),
            limit=args.candidate_limit,
            query_filter=query_filter,
        )

        title_seconds = (
            time.perf_counter()
            - started
        )

        started = time.perf_counter()

        bm25_points = store.query_bm25_points(
            collection_name=bm25_collection,
            query_text=query,
            limit=args.candidate_limit,
            query_filter=query_filter,
        )

        bm25_seconds = (
            time.perf_counter()
            - started
        )

    finally:
        store.close()

    started = time.perf_counter()

    results = multi_source_rrf(
        ranked_sources={
            "abstract": abstract_points,
            "title": title_points,
            "bm25": bm25_points,
        },
        rrf_k=args.rrf_k,
        limit=args.limit,
    )

    fusion_seconds = (
        time.perf_counter()
        - started
    )

    print("\nAbstract sorgu : %.4f sn" % abstract_seconds)
    print("Title sorgu    : %.4f sn" % title_seconds)
    print("BM25 sorgu     : %.4f sn" % bm25_seconds)
    print("RRF fusion     : %.6f sn" % fusion_seconds)

    print("\n" + "=" * 90)
    print("3-YOLLU HYBRID SONUÇLAR")
    print("=" * 90)

    print("\nSorgu:", query)

    for final_rank, row in enumerate(
        results,
        start=1,
    ):
        payload = row.get(
            "payload",
            {},
        )

        ranks = row.get(
            "ranks",
            {},
        )

        scores = row.get(
            "scores",
            {},
        )

        print("\n" + "-" * 90)

        print(
            "%d. Makale ID=%s | RRF=%.6f"
            % (
                final_rank,
                row.get(
                    "article_id"
                ),
                float(
                    row.get(
                        "rrf_score",
                        0.0,
                    )
                ),
            )
        )

        print(
            "Abstract : rank=%s score=%s"
            % (
                value_or_dash(
                    ranks.get(
                        "abstract"
                    )
                ),
                score_or_dash(
                    scores.get(
                        "abstract"
                    )
                ),
            )
        )

        print(
            "Title    : rank=%s score=%s"
            % (
                value_or_dash(
                    ranks.get(
                        "title"
                    )
                ),
                score_or_dash(
                    scores.get(
                        "title"
                    )
                ),
            )
        )

        print(
            "BM25     : rank=%s score=%s"
            % (
                value_or_dash(
                    ranks.get(
                        "bm25"
                    )
                ),
                score_or_dash(
                    scores.get(
                        "bm25"
                    )
                ),
            )
        )

        print(
            "Kanallar :",
            ", ".join(
                row.get(
                    "matched_sources",
                    [],
                )
            ),
        )

        print(
            "Başlık   :",
            payload.get(
                "title_tr"
            ),
        )

        print(
            "Yıl      :",
            payload.get(
                "publication_year"
            ),
        )

        print(
            "Primary  :",
            payload.get(
                "primary_topic"
            ),
        )

        print(
            "Secondary:",
            payload.get(
                "secondary_topic"
            ),
        )

    print(
        "\nNot: RRF, BM25 ve dense skorlar "
        "kalibre edilmiş olasılık değildir."
    )


if __name__ == "__main__":
    main()
