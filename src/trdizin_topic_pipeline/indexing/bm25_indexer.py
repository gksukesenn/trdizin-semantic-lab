#!/usr/bin/env python3
"""50.000 makale başlığı + keyword alanını Qdrant BM25 sparse indeksine yükler."""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import requests


ROOT = Path(__file__).resolve().parents[3]

from trdizin_topic_pipeline.search.qdrant_store import QdrantRestStore
from .helpers import atomic_json, point_id, read_articles, read_json, subject_names
from .payloads import build_bm25_text as bm25_text


ARTICLES_PATH = (
    ROOT
    / "data"
    / "processed"
    / "final_articles_50000.jsonl"
)

QUALITY_PATH = (
    ROOT
    / "outputs"
    / "final_50k"
    / "reports"
    / "dataset_quality_summary.json"
)

PROGRESS_PATH = (
    ROOT
    / "outputs"
    / "final_50k"
    / "search"
    / "qdrant_bm25_progress.json"
)

MANIFEST_PATH = (
    ROOT
    / "outputs"
    / "final_50k"
    / "search"
    / "qdrant_bm25_manifest.json"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/final_50k.json",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = ROOT / config_path

    config = read_json(config_path)
    qdrant = config.get("qdrant", {})

    url = str(
        qdrant.get(
            "url",
            "http://127.0.0.1:6335",
        )
    )

    collection = "trdizin_bm25_50000"

    expected_count = int(
        config.get(
            "target_article_count",
            50000,
        )
    )

    dataset_hash = str(
        read_json(
            QUALITY_PATH
        ).get(
            "dataset_sha256",
            "",
        )
    )

    articles = read_articles(ARTICLES_PATH)

    if len(articles) != expected_count:
        raise RuntimeError(
            "Makale sayısı yanlış: %d"
            % len(articles)
        )

    print("=" * 80)
    print("QDRANT 50.000 BM25 İNDEKSLEME")
    print("=" * 80)

    print("\nQdrant URL       :", url)
    print("Collection       :", collection)
    print("Makale           :", len(articles))
    print("Batch size       :", args.batch_size)
    print("Metin            : title_tr + keywords_tr")
    print("Dataset SHA-256  :", dataset_hash)

    store = QdrantRestStore(
        base_url=url,
        timeout_seconds=180,
    )

    session = requests.Session()

    try:
        exists = store.collection_exists(
            collection
        )

        if args.recreate and exists:
            print(
                "\nMevcut BM25 collection siliniyor..."
            )

            store.delete_collection(
                collection
            )

            exists = False

            if PROGRESS_PATH.exists():
                PROGRESS_PATH.unlink()

            if MANIFEST_PATH.exists():
                MANIFEST_PATH.unlink()

        if not exists:
            print(
                "\nBM25 collection oluşturuluyor..."
            )

            response = session.put(
                url
                + "/collections/"
                + collection,
                json={
                    "vectors": {},
                    "sparse_vectors": {
                        "text_bm25": {
                            "modifier": "idf"
                        }
                    },
                    "on_disk_payload": True,
                },
                timeout=180,
            )

            response.raise_for_status()

            start_row = 0

        else:
            current_count = store.exact_count(
                collection
            )

            print(
                "\nMevcut exact point:",
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
                    manifest.get(
                        "dataset_sha256"
                    )
                    == dataset_hash
                ):
                    print(
                        "Geçerli 50.000 BM25 point mevcut; "
                        "yeniden yüklenmedi."
                    )

                    return

            if PROGRESS_PATH.exists():
                progress = read_json(
                    PROGRESS_PATH
                )

                if (
                    progress.get(
                        "dataset_sha256"
                    )
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
                        "Qdrant count ile progress uyuşmuyor."
                    )

                print(
                    "Kaldığı yerden devam:",
                    start_row,
                )

            elif current_count == 0:
                start_row = 0

            else:
                raise RuntimeError(
                    "Kısmi collection var fakat progress yok. "
                    "--recreate kullanın."
                )

        started = time.time()

        for batch_start in range(
            start_row,
            expected_count,
            args.batch_size,
        ):
            batch_end = min(
                batch_start + args.batch_size,
                expected_count,
            )

            points = []

            for row_index in range(
                batch_start,
                batch_end,
            ):
                article = articles[
                    row_index
                ]

                article_id = str(
                    article["article_id"]
                )

                text = bm25_text(
                    article
                )

                points.append(
                    {
                        "id": point_id(
                            article_id
                        ),
                        "vector": {
                            "text_bm25": {
                                "text": text,
                                "model": "qdrant/bm25",
                                "options": {
                                    "language": "none",
                                    "tokenizer": "multilingual",
                                },
                            }
                        },
                        "payload": {
                            "row_index": row_index,
                            "article_id": article_id,
                            "title_tr": article.get(
                                "title_tr",
                                "",
                            ),
                            "publication_year": article.get(
                                "publication_year"
                            ),
                            "keywords_tr": article.get(
                                "keywords_tr",
                                [],
                            ),
                            "databases": article.get(
                                "databases",
                                [],
                            ),
                            "subjects": subject_names(
                                article
                            ),
                        },
                    }
                )

            response = session.put(
                url
                + "/collections/"
                + collection
                + "/points",
                params={"wait": "true"},
                json={"points": points},
                timeout=180,
            )

            response.raise_for_status()

            atomic_json(
                PROGRESS_PATH,
                {
                    "collection_name": collection,
                    "dataset_sha256": dataset_hash,
                    "completed_rows": batch_end,
                    "target_rows": expected_count,
                },
            )

            print(
                "BM25 ilerleme: %d/%d | geçen %.1f sn"
                % (
                    batch_end,
                    expected_count,
                    time.time() - started,
                )
            )

        final_count = store.exact_count(
            collection
        )

        if final_count != expected_count:
            raise RuntimeError(
                "Final count yanlış: %d"
                % final_count
            )

        atomic_json(
            MANIFEST_PATH,
            {
                "qdrant_url": url,
                "collection_name": collection,
                "point_count": final_count,
                "dataset_sha256": dataset_hash,
                "vector_name": "text_bm25",
                "model": "qdrant/bm25",
                "language": "none",
                "tokenizer": "multilingual",
                "text_fields": [
                    "title_tr",
                    "keywords_tr",
                ],
            },
        )

        print("\n" + "=" * 80)
        print("BM25 İNDEKSLEME TAMAMLANDI")
        print("=" * 80)

        print(
            "\nExact point:",
            final_count,
        )

    finally:
        session.close()
        store.close()


if __name__ == "__main__":
    main()
