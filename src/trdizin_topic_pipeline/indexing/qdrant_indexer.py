#!/usr/bin/env python3
"""50.000 makaleyi Qdrant collection içine yükler."""

import argparse
import csv
import json
import os
import sys
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[3]

from trdizin_topic_pipeline.search.qdrant_store import (
    QdrantRestStore,
)
from .helpers import atomic_json, point_id as qdrant_point_id, read_articles as read_jsonl, read_assignments, read_json
from .payloads import build_abstract_payload as build_payload
from .validation import validate_collection as validate_collection_vector_config, validate_vector_inputs

ARTICLES_PATH = (
    ROOT
    / "data"
    / "processed"
    / "final_articles_50000.jsonl"
)

EMBEDDINGS_PATH = (
    ROOT
    / "outputs"
    / "final_50k"
    / "embeddings"
    / "tr_mteb_50000.npy"
)

EMBEDDING_METADATA_PATH = (
    ROOT
    / "outputs"
    / "final_50k"
    / "embeddings"
    / "tr_mteb_50000_metadata.json"
)

QUALITY_SUMMARY_PATH = (
    ROOT
    / "outputs"
    / "final_50k"
    / "reports"
    / "dataset_quality_summary.json"
)

ASSIGNMENTS_PATH = (
    ROOT
    / "outputs"
    / "final_50k"
    / "clustering"
    / "final_cluster_assignments.csv"
)

PROGRESS_PATH = (
    ROOT
    / "outputs"
    / "final_50k"
    / "search"
    / "qdrant_index_progress.json"
)

MANIFEST_PATH = (
    ROOT
    / "outputs"
    / "final_50k"
    / "search"
    / "qdrant_index_manifest.json"
)

validate_inputs = partial(
    validate_vector_inputs, quality_path=QUALITY_SUMMARY_PATH,
    metadata_path=EMBEDDING_METADATA_PATH, vector_label="Embedding",
    require_nonempty_id=True,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/final_50k.json",
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "Mevcut collectionı silip sıfırdan oluşturur."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Config içindeki batch size değerini ezer.",
    )

    return parser.parse_args()


def main() -> None:
    args = arguments()

    config_path = (
        ROOT / args.config
        if not Path(args.config).is_absolute()
        else Path(args.config)
    )

    config = read_json(config_path)
    qdrant_config = config.get(
        "qdrant",
        {},
    )

    expected_count = int(
        config.get(
            "target_article_count",
            50000,
        )
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

    vector_size = int(
        qdrant_config.get(
            "vector_size",
            768,
        )
    )

    distance = str(
        qdrant_config.get(
            "distance",
            "Cosine",
        )
    )

    batch_size = int(
        args.batch_size
        or qdrant_config.get(
            "batch_size",
            128,
        )
    )

    timeout_seconds = int(
        qdrant_config.get(
            "timeout_seconds",
            180,
        )
    )

    print("=" * 80)
    print("QDRANT 50.000 MAKALE İNDEKSLEME")
    print("=" * 80)

    print("\nQdrant URL       :", qdrant_url)
    print("Collection       :", collection_name)
    print("Vector boyutu    :", vector_size)
    print("Mesafe           :", distance)
    print("Batch size       :", batch_size)

    articles = read_jsonl(
        ARTICLES_PATH
    )

    assignments = read_assignments(
        ASSIGNMENTS_PATH
    )

    embeddings = np.load(
        EMBEDDINGS_PATH,
        mmap_mode="r",
    )

    dataset_hash = validate_inputs(
        articles=articles,
        assignments=assignments,
        embeddings=embeddings,
        expected_count=expected_count,
    )

    print("\nGirdi doğrulaması başarılı.")
    print("Dataset SHA-256  :", dataset_hash)

    store = QdrantRestStore(
        base_url=qdrant_url,
        timeout_seconds=timeout_seconds,
    )

    try:
        exists = store.collection_exists(
            collection_name
        )

        if args.recreate and exists:
            print(
                "\nMevcut collection siliniyor..."
            )

            store.delete_collection(
                collection_name
            )

            exists = False

            if PROGRESS_PATH.exists():
                PROGRESS_PATH.unlink()

            if MANIFEST_PATH.exists():
                MANIFEST_PATH.unlink()

        if not exists:
            print(
                "\nCollection oluşturuluyor..."
            )

            store.create_collection(
                collection_name=collection_name,
                vector_size=vector_size,
                distance=distance,
            )

            start_row = 0
        else:
            info = store.collection_info(
                collection_name
            )

            validate_collection_vector_config(
                info=info,
                expected_size=vector_size,
                expected_distance=distance,
            )

            current_count = store.exact_count(
                collection_name
            )

            print(
                "\nMevcut exact point sayısı:",
                current_count,
            )

            if (
                current_count == expected_count
                and MANIFEST_PATH.exists()
            ):
                manifest = read_json(
                    MANIFEST_PATH
                )

                if (
                    manifest.get("dataset_sha256")
                    == dataset_hash
                ):
                    print(
                        "Geçerli 50.000 point mevcut; "
                        "vektörler yeniden yüklenmeyecek."
                    )

                    start_row = expected_count
                else:
                    raise RuntimeError(
                        "Collection dolu fakat dataset "
                        "hash eşleşmiyor. Güvenli yeniden "
                        "oluşturmak için --recreate kullanın."
                    )

            elif current_count == 0:
                start_row = 0

            elif PROGRESS_PATH.exists():
                progress = read_json(
                    PROGRESS_PATH
                )

                if (
                    progress.get("dataset_sha256")
                    != dataset_hash
                ):
                    raise RuntimeError(
                        "Progress dataset hash eşleşmiyor."
                    )

                start_row = int(
                    progress.get(
                        "completed_rows",
                        0,
                    )
                )

                if current_count != start_row:
                    raise RuntimeError(
                        "Qdrant count ile progress uyuşmuyor.\n"
                        "Qdrant : %d\n"
                        "Progress: %d"
                        % (
                            current_count,
                            start_row,
                        )
                    )

                print(
                    "Yükleme kaldığı yerden devam edecek:",
                    start_row,
                )

            else:
                raise RuntimeError(
                    "Collection kısmen dolu fakat geçerli "
                    "progress dosyası yok. Güvenli sıfırlama "
                    "için --recreate kullanın."
                )

        started = time.time()

        for batch_start in range(
            start_row,
            expected_count,
            batch_size,
        ):
            batch_end = min(
                batch_start + batch_size,
                expected_count,
            )

            points: List[
                Dict[str, Any]
            ] = []

            for row_index in range(
                batch_start,
                batch_end,
            ):
                article = articles[
                    row_index
                ]

                assignment = assignments[
                    row_index
                ]

                article_id = str(
                    article["article_id"]
                )

                points.append(
                    {
                        "id": qdrant_point_id(
                            article_id
                        ),
                        "vector": embeddings[
                            row_index
                        ].tolist(),
                        "payload": build_payload(
                            row_index=row_index,
                            article=article,
                            assignment=assignment,
                        ),
                    }
                )

            store.upsert_points(
                collection_name=collection_name,
                points=points,
            )

            atomic_json(
                PROGRESS_PATH,
                {
                    "collection_name": (
                        collection_name
                    ),
                    "dataset_sha256": (
                        dataset_hash
                    ),
                    "completed_rows": (
                        batch_end
                    ),
                    "target_rows": (
                        expected_count
                    ),
                    "batch_size": (
                        batch_size
                    ),
                },
            )

            elapsed = time.time() - started

            print(
                "Qdrant ilerleme: %d/%d | geçen %.1f sn"
                % (
                    batch_end,
                    expected_count,
                    elapsed,
                )
            )

        final_count = store.exact_count(
            collection_name
        )

        if final_count != expected_count:
            raise RuntimeError(
                "Final exact count yanlış: %d"
                % final_count
            )

        print(
            "\nPayload indexleri oluşturuluyor..."
        )

        payload_indexes = [
            ("publication_year", "integer"),
            ("databases", "keyword"),
            ("subjects", "keyword"),
            ("assignment_method", "keyword"),
            ("primary_cluster", "integer"),
            ("secondary_cluster", "integer"),
            ("primary_topic", "keyword"),
            ("secondary_topic", "keyword"),
        ]

        for (
            field_name,
            field_schema,
        ) in payload_indexes:
            store.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )

            print(
                "- %s (%s)"
                % (
                    field_name,
                    field_schema,
                )
            )

        manifest = {
            "qdrant_url": qdrant_url,
            "collection_name": (
                collection_name
            ),
            "point_count": final_count,
            "vector_size": vector_size,
            "distance": distance,
            "dataset_sha256": dataset_hash,
            "embedding_path": str(
                EMBEDDINGS_PATH.relative_to(
                    ROOT
                )
            ),
            "articles_path": str(
                ARTICLES_PATH.relative_to(
                    ROOT
                )
            ),
            "assignments_path": str(
                ASSIGNMENTS_PATH.relative_to(
                    ROOT
                )
            ),
            "payload_indexes": [
                {
                    "field_name": field_name,
                    "field_schema": field_schema,
                }
                for (
                    field_name,
                    field_schema,
                ) in payload_indexes
            ],
        }

        atomic_json(
            MANIFEST_PATH,
            manifest,
        )

        print("\n" + "=" * 80)
        print("QDRANT İNDEKSLEME TAMAMLANDI")
        print("=" * 80)

        print(
            "\nExact point sayısı:",
            final_count,
        )

        print(
            "Manifest           :",
            MANIFEST_PATH,
        )

    finally:
        store.close()


if __name__ == "__main__":
    main()
