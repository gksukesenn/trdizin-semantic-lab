#!/usr/bin/env python3
"""Türkçe sorguları TR-MTEB + Qdrant ile semantik olarak arar."""

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

from trdizin_topic_pipeline.search.qdrant_store import (
    QdrantRestStore,
)
from trdizin_topic_pipeline.search.filters import build_filter
from trdizin_topic_pipeline.search.cli_support import (
    encode_cli_query as encode_query, load_cli_model as load_model, read_json,
    resolve_path as _resolve_path, select_cli_device as select_device, shorten,
)

resolve_config_path = partial(_resolve_path, ROOT)


DEFAULT_MODEL_ID = (
    "trmteb/turkish-embedding-model-fine-tuned"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "50.000 Türkçe makale içinde "
            "TR-MTEB tabanlı semantic search yapar."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/final_50k.json",
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Aranacak Türkçe metin.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=10,
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
        help="Örneğin SCIENCE veya SOCIAL.",
    )

    parser.add_argument(
        "--assignment-method",
        help=(
            "Örneğin centroid_fallback veya "
            "doğrudan HDBSCAN atama değeri."
        ),
    )

    parser.add_argument(
        "--primary-topic",
        help="Tam eşleşen primary_topic filtresi.",
    )

    parser.add_argument(
        "--show-abstract",
        action="store_true",
    )

    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help=(
            "CUDA bulunamazsa CPU üzerinde çalışmaya izin verir."
        ),
    )

    return parser.parse_args()


def print_device_information(
    device: str,
) -> None:
    print("=" * 80)
    print("TR-MTEB SEMANTIC SEARCH")
    print("=" * 80)

    print("\nPython executable         :", sys.executable)
    print("Python version            :", sys.version.split()[0])
    print("torch version             :", torch.__version__)
    print("torch CUDA build          :", torch.version.cuda)
    print(
        "CUDA_VISIBLE_DEVICES      :",
        os.environ.get(
            "CUDA_VISIBLE_DEVICES",
            "<ayarlanmamış>",
        ),
    )
    print(
        "torch.cuda.is_available() :",
        torch.cuda.is_available(),
    )
    print(
        "Seçilen cihaz             :",
        device,
    )

    if device == "cuda":
        print(
            "GPU                       :",
            torch.cuda.get_device_name(0),
        )


def print_results(
    query: str,
    points: List[Dict[str, Any]],
    show_abstract: bool,
) -> None:
    print("\n" + "=" * 80)
    print("SEMANTIC SEARCH SONUÇLARI")
    print("=" * 80)

    print("\nSorgu:", query)
    print("Sonuç:", len(points))

    if not points:
        print(
            "\nFiltrelere uyan sonuç bulunamadı."
        )
        return

    for rank, point in enumerate(
        points,
        start=1,
    ):
        payload = point.get(
            "payload",
            {},
        )

        score = float(
            point.get(
                "score",
                0.0,
            )
        )

        print("\n" + "-" * 80)
        print(
            "%d. Qdrant ID=%s | semantic score=%.4f"
            % (
                rank,
                point.get("id"),
                score,
            )
        )

        print(
            "Makale ID     :",
            payload.get("article_id"),
        )

        print(
            "Başlık        :",
            payload.get("title_tr"),
        )

        print(
            "Yıl           :",
            payload.get(
                "publication_year",
                "Bilinmiyor",
            ),
        )

        print(
            "Database      :",
            ", ".join(
                payload.get(
                    "databases",
                    [],
                )
            ),
        )

        print(
            "Atama yöntemi :",
            payload.get(
                "assignment_method",
            ),
        )

        print(
            "Birincil konu :",
            payload.get(
                "primary_topic",
            ),
        )

        print(
            "İkincil konu  :",
            payload.get(
                "secondary_topic",
            ),
        )

        margin = payload.get(
            "similarity_margin"
        )

        if margin is not None:
            print(
                "Konu marjı    : %.4f"
                % float(margin)
            )

        if show_abstract:
            print(
                "Abstract      :",
                shorten(
                    payload.get(
                        "abstract_tr",
                        "",
                    ),
                    maximum_length=700,
                ),
            )

    print(
        "\nNot: Qdrant semantic score ve konu marjı "
        "kalibre edilmiş olasılık değildir."
    )


def main() -> None:
    args = arguments()

    if not args.query.strip():
        raise ValueError(
            "Sorgu boş olamaz."
        )

    if args.limit < 1 or args.limit > 100:
        raise ValueError(
            "--limit 1 ile 100 arasında olmalıdır."
        )

    config = read_json(
        resolve_config_path(
            args.config
        )
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
            DEFAULT_MODEL_ID,
        )
    )

    device = select_device(
        allow_cpu=args.allow_cpu
    )

    print_device_information(
        device
    )

    model = load_model(
        model_id=model_id,
        device=device,
    )

    query_vector = encode_query(
        model=model,
        query=args.query.strip(),
        device=device,
    )

    query_filter = build_filter(
        year_from=args.year_from,
        year_to=args.year_to,
        database=args.database,
        assignment_method=(
            args.assignment_method
        ),
        primary_topic=args.primary_topic,
    )

    print("\nQdrant URL                :", qdrant_url)
    print("Collection                :", collection_name)
    print(
        "Filtre                     :",
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
        started = time.perf_counter()

        points = store.query_points(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=args.limit,
            query_filter=query_filter,
        )

        search_elapsed = (
            time.perf_counter()
            - started
        )

        print(
            "Qdrant sorgu süresi       : %.4f sn"
            % search_elapsed
        )

    finally:
        store.close()

    print_results(
        query=args.query.strip(),
        points=points,
        show_abstract=args.show_abstract,
    )


if __name__ == "__main__":
    main()
