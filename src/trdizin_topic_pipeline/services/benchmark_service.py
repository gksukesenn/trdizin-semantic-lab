#!/usr/bin/env python3
"""Qdrant semantic-search sonuçlarını metadata ile keşifsel olarak ölçer."""

import argparse
import csv
import json
import math
import os
import sys
import time
from functools import partial
import unicodedata
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[3]

from trdizin_topic_pipeline.search.qdrant_store import (
    QdrantRestStore,
)
from trdizin_topic_pipeline.evaluation.retrieval import (
    is_metadata_relevant, ndcg_at_k, precision_at_k, reciprocal_rank,
)
from trdizin_topic_pipeline.reporting.retrieval_report import (
    write_markdown, write_results_csv, write_summary_csv,
)
from trdizin_topic_pipeline.search.cli_support import (
    encode_queries, load_cli_model as load_model, read_json,
    resolve_path as _resolve_path, select_cli_device as select_device,
)

resolve_path = partial(_resolve_path, ROOT)


DEFAULT_QUERIES_PATH = (
    ROOT
    / "configs"
    / "retrieval_benchmark_queries.json"
)

OUTPUT_DIRECTORY = (
    ROOT
    / "outputs"
    / "final_50k"
    / "search"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/final_50k.json",
    )

    parser.add_argument(
        "--queries",
        default=str(DEFAULT_QUERIES_PATH),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--allow-cpu",
        action="store_true",
    )

    return parser.parse_args()


def write_json(
    path: Path,
    value: Dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = arguments()

    if args.limit < 10:
        raise ValueError(
            "Benchmark için --limit en az 10 olmalıdır."
        )

    config = read_json(
        resolve_path(args.config)
    )

    benchmark_config = read_json(
        resolve_path(args.queries)
    )

    queries = benchmark_config.get(
        "queries",
        [],
    )

    if not isinstance(queries, list) or not queries:
        raise ValueError(
            "Benchmark sorguları bulunamadı."
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

    collection_name = str(
        qdrant_config.get(
            "collection_name",
            "trdizin_articles_50000",
        )
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

    print("=" * 80)
    print("50.000 MAKALE RETRIEVAL BENCHMARKI")
    print("=" * 80)

    print("\nSorgu sayısı       :", len(queries))
    print("Sonuç limiti       :", args.limit)
    print("Qdrant URL         :", qdrant_url)
    print("Collection         :", collection_name)
    print("Cihaz              :", device)

    if device == "cuda":
        print(
            "GPU                :",
            torch.cuda.get_device_name(0),
        )

    model = load_model(
        model_id=model_id,
        device=device,
        style="benchmark",
    )

    query_texts = [
        str(row["query"])
        for row in queries
    ]

    embeddings = encode_queries(
        model=model,
        query_texts=query_texts,
        device=device,
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

    result_rows: List[
        Dict[str, Any]
    ] = []

    summary_rows: List[
        Dict[str, Any]
    ] = []

    try:
        for query_index, query_config in enumerate(
            queries
        ):
            query_id = str(
                query_config["query_id"]
            )

            query = str(
                query_config["query"]
            )

            relevance_groups = (
                query_config.get(
                    "relevance_groups",
                    [],
                )
            )

            started = time.perf_counter()

            points = store.query_points(
                collection_name=collection_name,
                query_vector=embeddings[
                    query_index
                ].tolist(),
                limit=args.limit,
            )

            elapsed = (
                time.perf_counter()
                - started
            )

            relevance: List[int] = []

            for rank, point in enumerate(
                points,
                start=1,
            ):
                payload = point.get(
                    "payload",
                    {},
                )

                relevant = is_metadata_relevant(
                    payload=payload,
                    relevance_groups=relevance_groups,
                )

                relevance.append(
                    int(relevant)
                )

                result_rows.append(
                    {
                        "query_id": query_id,
                        "query": query,
                        "rank": rank,
                        "qdrant_id": point.get("id"),
                        "article_id": payload.get(
                            "article_id"
                        ),
                        "semantic_score": float(
                            point.get(
                                "score",
                                0.0,
                            )
                        ),
                        "metadata_relevant": int(
                            relevant
                        ),
                        "title_tr": payload.get(
                            "title_tr",
                            "",
                        ),
                        "publication_year": payload.get(
                            "publication_year",
                            "",
                        ),
                        "databases": "|".join(
                            payload.get(
                                "databases",
                                [],
                            )
                        ),
                        "assignment_method": payload.get(
                            "assignment_method",
                            "",
                        ),
                        "primary_cluster": payload.get(
                            "primary_cluster",
                            "",
                        ),
                        "primary_topic": payload.get(
                            "primary_topic",
                            "",
                        ),
                        "secondary_cluster": payload.get(
                            "secondary_cluster",
                            "",
                        ),
                        "secondary_topic": payload.get(
                            "secondary_topic",
                            "",
                        ),
                        "similarity_margin": payload.get(
                            "similarity_margin",
                            "",
                        ),
                        "human_judgment": "",
                    }
                )

            summary = {
                "query_id": query_id,
                "query": query,
                "result_count": len(points),
                "precision_at_5": precision_at_k(
                    relevance,
                    5,
                ),
                "precision_at_10": precision_at_k(
                    relevance,
                    10,
                ),
                "mrr_at_10": reciprocal_rank(
                    relevance[:10]
                ),
                "ndcg_at_10": ndcg_at_k(
                    relevance,
                    10,
                ),
                "qdrant_seconds": elapsed,
            }

            summary_rows.append(
                summary
            )

            print(
                "%s | P@5=%.2f | P@10=%.2f | "
                "MRR=%.2f | nDCG=%.2f | %.4f sn"
                % (
                    query_id,
                    summary["precision_at_5"],
                    summary["precision_at_10"],
                    summary["mrr_at_10"],
                    summary["ndcg_at_10"],
                    elapsed,
                )
            )

    finally:
        store.close()

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        OUTPUT_DIRECTORY
        / "retrieval_benchmark_results.csv"
    )

    summary_csv_path = (
        OUTPUT_DIRECTORY
        / "retrieval_benchmark_summary.csv"
    )

    summary_json_path = (
        OUTPUT_DIRECTORY
        / "retrieval_benchmark_summary.json"
    )

    report_path = (
        OUTPUT_DIRECTORY
        / "RETRIEVAL_BENCHMARK_REPORT.md"
    )

    write_results_csv(
        result_rows,
        results_path,
    )

    write_summary_csv(
        summary_rows,
        summary_csv_path,
    )

    write_json(
        summary_json_path,
        {
            "query_count": len(queries),
            "limit": args.limit,
            "average_precision_at_5": mean(
                float(row["precision_at_5"])
                for row in summary_rows
            ),
            "average_precision_at_10": mean(
                float(row["precision_at_10"])
                for row in summary_rows
            ),
            "average_mrr_at_10": mean(
                float(row["mrr_at_10"])
                for row in summary_rows
            ),
            "average_ndcg_at_10": mean(
                float(row["ndcg_at_10"])
                for row in summary_rows
            ),
            "queries": summary_rows,
            "evaluation_type": (
                "TR Dizin subject metadata tabanlı "
                "keşifsel retrieval tutarlılığı"
            ),
        },
    )

    write_markdown(
        summary_rows,
        report_path,
    )

    print("\n" + "=" * 80)
    print("BENCHMARK TAMAMLANDI")
    print("=" * 80)

    print("\nAyrıntılı sonuçlar :", results_path)
    print("Özet CSV           :", summary_csv_path)
    print("Özet JSON          :", summary_json_path)
    print("Okunabilir rapor   :", report_path)


if __name__ == "__main__":
    main()
