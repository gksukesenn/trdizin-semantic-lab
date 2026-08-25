#!/usr/bin/env python3
"""50.000 makale başlığı için normalize TR-MTEB embeddingi üretir."""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[3]
from trdizin_topic_pipeline.search.cli_support import read_json, select_cli_device as select_device

ARTICLES_PATH = (
    ROOT
    / "data"
    / "processed"
    / "final_articles_50000.jsonl"
)

OUTPUT_PATH = (
    ROOT
    / "outputs"
    / "final_50k"
    / "embeddings"
    / "tr_mteb_titles_50000.npy"
)

METADATA_PATH = (
    ROOT
    / "outputs"
    / "final_50k"
    / "embeddings"
    / "tr_mteb_titles_50000_metadata.json"
)

QUALITY_SUMMARY_PATH = (
    ROOT
    / "outputs"
    / "final_50k"
    / "reports"
    / "dataset_quality_summary.json"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/final_50k.json",
    )

    parser.add_argument(
        "--allow-cpu",
        action="store_true",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    return parser.parse_args()


def read_titles(path: Path) -> List[str]:
    titles: List[str] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            row = json.loads(line)

            title = " ".join(
                str(
                    row.get(
                        "title_tr",
                        "",
                    )
                ).split()
            )

            if not title:
                raise ValueError(
                    "Boş başlık bulundu: satır %d"
                    % line_number
                )

            titles.append(title)

    return titles


def atomic_json(
    path: Path,
    value: Dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        str(temporary),
        str(path),
    )


def main() -> None:
    args = arguments()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = ROOT / config_path

    config = read_json(config_path)

    expected_count = int(
        config.get(
            "target_article_count",
            50000,
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

    dataset_hash = str(
        read_json(
            QUALITY_SUMMARY_PATH
        ).get(
            "dataset_sha256",
            "",
        )
    )

    if not dataset_hash:
        raise RuntimeError(
            "Dataset SHA-256 bulunamadı."
        )

    titles = read_titles(
        ARTICLES_PATH
    )

    if len(titles) != expected_count:
        raise RuntimeError(
            "Başlık sayısı yanlış: %d"
            % len(titles)
        )

    if OUTPUT_PATH.exists() and METADATA_PATH.exists():
        metadata = read_json(
            METADATA_PATH
        )

        if (
            metadata.get("dataset_sha256")
            == dataset_hash
        ):
            values = np.load(
                OUTPUT_PATH,
                mmap_mode="r",
            )

            if values.shape == (
                expected_count,
                768,
            ):
                print(
                    "Geçerli başlık embeddingi mevcut; "
                    "yeniden üretilmedi."
                )
                return

    device = select_device(
        args.allow_cpu
    )

    print("=" * 80)
    print("50.000 BAŞLIK EMBEDDINGİ")
    print("=" * 80)

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

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = SentenceTransformer(
        model_id,
        device=device,
        trust_remote_code=False,
    )

    model.max_seq_length = 512

    parameter_device = next(
        model.parameters()
    ).device

    print("Model                     :", model_id)
    print("model.device              :", model.device)
    print("İlk parametre cihazı      :", parameter_device)
    print("Batch size                :", args.batch_size)

    if str(model.device).split(":")[0] != device:
        raise RuntimeError(
            "Model beklenen cihazda değil."
        )

    if (
        str(parameter_device).split(":")[0]
        != device
    ):
        raise RuntimeError(
            "Model parametreleri beklenen cihazda değil."
        )

    started = time.perf_counter()

    values = model.encode(
        titles,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - started

    values = values.astype(
        np.float32,
        copy=False,
    )

    if values.shape != (
        expected_count,
        768,
    ):
        raise RuntimeError(
            "Başlık embedding şekli yanlış: %r"
            % (values.shape,)
        )

    if not np.isfinite(values).all():
        raise RuntimeError(
            "Başlık embeddingi NaN/Inf içeriyor."
        )

    norms = np.linalg.norm(
        values,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-4,
    ):
        raise RuntimeError(
            "Başlık embeddingleri normalize değil."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = OUTPUT_PATH.with_suffix(
        ".npy.tmp"
    )

    with temporary_path.open("wb") as handle:
        np.save(
            handle,
            values,
            allow_pickle=False,
        )

        handle.flush()
        os.fsync(handle.fileno())

    os.replace(
        str(temporary_path),
        str(OUTPUT_PATH),
    )

    metadata = {
        "model_id": model_id,
        "text_field": "title_tr",
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "normalize_embeddings": True,
        "dataset_sha256": dataset_hash,
        "device": str(model.device),
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "mean_norm": float(norms.mean()),
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
    }

    if device == "cuda":
        metadata[
            "max_cuda_memory_bytes"
        ] = int(
            torch.cuda.max_memory_allocated()
        )

    atomic_json(
        METADATA_PATH,
        metadata,
    )

    print("\nBaşlık embeddingi tamamlandı.")
    print("Şekil                     :", values.shape)
    print("Süre                      : %.2f sn" % elapsed)
    print("Dosya                     :", OUTPUT_PATH)
    print("Metadata                  :", METADATA_PATH)


if __name__ == "__main__":
    main()
