import csv
import gc
import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# 1. Bu aşamada kullanacağımız finalist modeller
# ---------------------------------------------------------
#
# input_field:
# Embedding'e yalnızca Türkçe abstract veriyoruz.
#
# Subject, title ve keywords bu aşamada modele verilmez.
# Böylece modeller tamamen aynı girdiyi görür.
#

MODEL_CONFIGS: List[Dict[str, Any]] = [
    {
        "short_name": "TR-MTEB",
        "model_id": (
            "trmteb/"
            "turkish-embedding-model-fine-tuned"
        ),
        "prefix": "",
        "max_seq_length": 512,
        "trust_remote_code": False,
        "output_filename": "tr_mteb.npy",
    },
    {
        "short_name": "E5-large",
        "model_id": (
            "intfloat/"
            "multilingual-e5-large"
        ),
        # E5 model kartında clustering için öneriliyor.
        "prefix": "query: ",
        "max_seq_length": 512,
        "trust_remote_code": False,
        "output_filename": "e5_large.npy",
    },
    {
        "short_name": "GTE-multilingual",
        "model_id": (
            "Alibaba-NLP/"
            "gte-multilingual-base"
        ),
        "prefix": "",
        "max_seq_length": 8192,
        "trust_remote_code": True,
        "output_filename": "gte_multilingual.npy",
    },
]


# Day 10 deneyinde batch_size=4 bütün modellerde çalışmıştı.
# Bellek yetmezse sırasıyla 2 ve 1'e düşeceğiz.
BATCH_SIZE_CANDIDATES = [4, 2, 1]

INPUT_FIELD = "abstract_tr"


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


def get_dataset_path() -> Path:
    """1.000 makalelik pilot veri dosyasının yolunu döndürür."""

    return (
        get_project_root()
        / "data"
        / "processed"
        / "pilot_articles_1000.jsonl"
    )


def calculate_file_sha256(file_path: Path) -> str:
    """
    Veri dosyasının SHA-256 kimliğini hesaplar.

    Bu değer, embeddinglerin tam olarak hangi veri
    dosyasından üretildiğini doğrulamamızı sağlar.
    """

    sha256 = hashlib.sha256()

    with file_path.open("rb") as input_file:
        while True:
            chunk = input_file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def load_articles() -> Tuple[List[Dict[str, Any]], Path, str]:
    """
    Pilot JSONL dosyasını okur ve temel kontrolleri yapar.

    Makalelerin dosyadaki sırası korunur.
    Çünkü embedding matrisindeki satır sırası da
    aynı sıraya göre oluşacaktır.
    """

    dataset_path = get_dataset_path()

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Pilot veri dosyası bulunamadı:\n{dataset_path}"
        )

    articles: List[Dict[str, Any]] = []
    seen_article_ids = set()

    with dataset_path.open(
        mode="r",
        encoding="utf-8",
    ) as input_file:
        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            cleaned_line = line.strip()

            if not cleaned_line:
                continue

            try:
                article = json.loads(cleaned_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL satırı okunamadı: {line_number}"
                ) from error

            if not isinstance(article, dict):
                continue

            article_id = str(
                article.get("article_id", "")
            ).strip()

            abstract = article.get(INPUT_FIELD)

            if not article_id:
                raise ValueError(
                    f"Makale ID bulunamadı. Satır: {line_number}"
                )

            if article_id in seen_article_ids:
                raise ValueError(
                    f"Tekrarlı makale ID bulundu: {article_id}"
                )

            if not isinstance(abstract, str):
                raise ValueError(
                    f"Abstract string değil. "
                    f"ID={article_id}"
                )

            if not abstract.strip():
                raise ValueError(
                    f"Boş abstract bulundu. "
                    f"ID={article_id}"
                )

            seen_article_ids.add(article_id)
            articles.append(article)

    if not articles:
        raise ValueError(
            "Pilot veri dosyasında geçerli makale bulunamadı."
        )

    dataset_sha256 = calculate_file_sha256(
        dataset_path
    )

    return articles, dataset_path, dataset_sha256


def get_output_directory() -> Path:
    """Embedding çıktı klasörünü oluşturur."""

    output_directory = (
        get_project_root()
        / "research" / "outputs"
        / "day13_embeddings"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory


def synchronize_cuda() -> None:
    """CUDA işlemlerinin tamamlanmasını bekler."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_memory() -> None:
    """Python ve CUDA belleğinde kullanılmayan alanları temizler."""

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_peak_gpu_memory_mb() -> float:
    """Tepe CUDA bellek kullanımını MB olarak döndürür."""

    if not torch.cuda.is_available():
        return 0.0

    return (
        torch.cuda.max_memory_allocated()
        / 1024
        / 1024
    )


def sanitize_filename(value: str) -> str:
    """Metni güvenli dosya adına dönüştürür."""

    return re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        value,
    ).strip("_").lower()


def atomic_save_numpy(
    output_path: Path,
    embeddings: np.ndarray,
) -> None:
    """
    NumPy matrisini önce geçici dosyaya yazar.

    Program kayıt sırasında kapanırsa yarım kalmış ana
    embedding dosyasının oluşmasını önler.
    """

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with temporary_path.open("wb") as output_file:
        np.save(
            output_file,
            embeddings,
        )

    os.replace(
        temporary_path,
        output_path,
    )


def encode_with_batch_fallback(
    model: SentenceTransformer,
    texts: List[str],
) -> Tuple[np.ndarray, int, float, float]:
    """
    Bütün metinleri embedding'e dönüştürür.

    GPU belleği yetmezse daha küçük batch size ile
    otomatik olarak tekrar dener.
    """

    last_error: Exception = RuntimeError(
        "Embedding üretilemedi."
    )

    for batch_size in BATCH_SIZE_CANDIDATES:
        try:
            print(
                f"\nEmbedding deneniyor: "
                f"batch_size={batch_size}"
            )

            clear_memory()

            # Küçük bir ısınma çalışması.
            # İlk CUDA hazırlık süresini ölçümden ayırır.
            model.encode(
                texts[:2],
                batch_size=min(batch_size, 2),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            synchronize_cuda()

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            encoding_start = time.perf_counter()

            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )

            synchronize_cuda()

            encoding_seconds = (
                time.perf_counter()
                - encoding_start
            )

            peak_gpu_memory_mb = (
                get_peak_gpu_memory_mb()
            )

            return (
                embeddings,
                batch_size,
                encoding_seconds,
                peak_gpu_memory_mb,
            )

        except RuntimeError as error:
            last_error = error

            error_message = str(error).lower()

            if "out of memory" not in error_message:
                raise

            print(
                f"GPU belleği yetmedi: "
                f"batch_size={batch_size}"
            )

            clear_memory()

    raise last_error


def create_article_index(
    articles: List[Dict[str, Any]],
    dataset_sha256: str,
) -> Path:
    """
    Embedding satır numarası ile makale bilgisini eşleştiren
    CSV dosyasını oluşturur.

    Örneğin:
    row_index=25 → embedding matrisinin 25. satırı
                 → belirli bir TR Dizin makalesi
    """

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day13_article_index.csv"
    )

    fieldnames = [
        "row_index",
        "article_id",
        "publication_year",
        "databases",
        "title_tr",
        "abstract_character_count",
        "subject_count",
        "dataset_sha256",
    ]

    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row_index, article in enumerate(articles):
            subjects = article.get("subjects")

            subject_count = (
                len(subjects)
                if isinstance(subjects, list)
                else 0
            )

            writer.writerow(
                {
                    "row_index": row_index,
                    "article_id": article.get(
                        "article_id",
                        "",
                    ),
                    "publication_year": article.get(
                        "publication_year",
                        "",
                    ),
                    "databases": json.dumps(
                        article.get(
                            "databases",
                            [],
                        ),
                        ensure_ascii=False,
                    ),
                    "title_tr": article.get(
                        "title_tr",
                        "",
                    ),
                    "abstract_character_count": len(
                        article[INPUT_FIELD]
                    ),
                    "subject_count": subject_count,
                    "dataset_sha256": dataset_sha256,
                }
            )

    return output_path


def benchmark_and_save_model(
    config: Dict[str, Any],
    articles: List[Dict[str, Any]],
    dataset_sha256: str,
) -> Dict[str, Any]:
    """Tek bir modelle 1.000 makalenin embeddingini üretir."""

    print("\n" + "=" * 80)
    print(f"MODEL: {config['short_name']}")
    print("=" * 80)

    print(f"\nModel kimliği : {config['model_id']}")
    print(f"Metin alanı   : {INPUT_FIELD}")
    print(f"Metin öneki   : {config['prefix']!r}")
    print(
        f"Token sınırı  : "
        f"{config['max_seq_length']}"
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Çalışma cihazı: {device}")
    print("\nModel yükleniyor...")

    synchronize_cuda()
    load_start = time.perf_counter()

    model = SentenceTransformer(
        config["model_id"],
        device=device,
        trust_remote_code=config[
            "trust_remote_code"
        ],
    )

    model.max_seq_length = int(
        config["max_seq_length"]
    )

    synchronize_cuda()

    load_seconds = (
        time.perf_counter()
        - load_start
    )

    texts = [
        config["prefix"] + article[INPUT_FIELD]
        for article in articles
    ]

    (
        embeddings,
        used_batch_size,
        encoding_seconds,
        peak_gpu_memory_mb,
    ) = encode_with_batch_fallback(
        model=model,
        texts=texts,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embedding matrisi iki boyutlu değil: "
            f"{embeddings.shape}"
        )

    if embeddings.shape[0] != len(articles):
        raise ValueError(
            "Embedding satır sayısı ile makale sayısı "
            "uyuşmuyor."
        )

    output_directory = get_output_directory()

    embedding_path = (
        output_directory
        / config["output_filename"]
    )

    atomic_save_numpy(
        output_path=embedding_path,
        embeddings=embeddings,
    )

    vector_file_mb = (
        embedding_path.stat().st_size
        / 1024
        / 1024
    )

    vector_norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    result = {
        "short_name": config["short_name"],
        "model_id": config["model_id"],
        "dataset_sha256": dataset_sha256,
        "input_field": INPUT_FIELD,
        "prefix": config["prefix"],
        "device": device,
        "max_seq_length": config[
            "max_seq_length"
        ],
        "article_count": len(articles),
        "batch_size": used_batch_size,
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "embedding_dtype": str(
            embeddings.dtype
        ),
        "load_seconds": load_seconds,
        "encoding_seconds": encoding_seconds,
        "articles_per_second": (
            len(articles)
            / encoding_seconds
        ),
        "peak_gpu_memory_mb": (
            peak_gpu_memory_mb
        ),
        "vector_file_mb": vector_file_mb,
        "minimum_vector_norm": float(
            np.min(vector_norms)
        ),
        "mean_vector_norm": float(
            np.mean(vector_norms)
        ),
        "maximum_vector_norm": float(
            np.max(vector_norms)
        ),
        "embedding_path": str(
            embedding_path
        ),
        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
    }

    metadata_path = (
        output_directory
        / (
            sanitize_filename(
                config["short_name"]
            )
            + "_metadata.json"
        )
    )

    with metadata_path.open(
        mode="w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            result,
            metadata_file,
            ensure_ascii=False,
            indent=2,
        )

    result["metadata_path"] = str(
        metadata_path
    )

    print("\nSonuç:")
    print(
        f"- Embedding şekli : "
        f"{embeddings.shape}"
    )
    print(
        f"- Yükleme süresi  : "
        f"{load_seconds:.2f} saniye"
    )
    print(
        f"- Encoding süresi : "
        f"{encoding_seconds:.2f} saniye"
    )
    print(
        f"- Makale/saniye   : "
        f"{result['articles_per_second']:.2f}"
    )
    print(
        f"- Tepe GPU belleği: "
        f"{peak_gpu_memory_mb:.2f} MB"
    )
    print(
        f"- Vektör dosyası  : "
        f"{vector_file_mb:.2f} MB"
    )
    print(
        f"- Kayıt yolu      : "
        f"{embedding_path}"
    )

    del embeddings
    del model

    clear_memory()

    return result


def save_summary(
    results: List[Dict[str, Any]],
) -> Tuple[Path, Path]:
    """Üç modelin özet sonuçlarını CSV ve JSON'a kaydeder."""

    output_directory = (
        get_project_root()
        / "research" / "outputs"
    )

    csv_path = (
        output_directory
        / "day13_embedding_summary.csv"
    )

    json_path = (
        output_directory
        / "day13_embedding_summary.json"
    )

    fieldnames = list(
        results[0].keys()
    )

    with csv_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    with json_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            results,
            json_file,
            ensure_ascii=False,
            indent=2,
        )

    return csv_path, json_path


def main() -> None:
    print("=" * 80)
    print("1.000 TÜRKÇE ABSTRACT EMBEDDING ÜRETİMİ")
    print("=" * 80)

    (
        articles,
        dataset_path,
        dataset_sha256,
    ) = load_articles()

    print(f"\nVeri dosyası:\n{dataset_path}")
    print(
        f"\nOkunan makale sayısı: "
        f"{len(articles)}"
    )
    print(
        f"Veri SHA-256 başlangıcı: "
        f"{dataset_sha256[:16]}..."
    )
    print(
        f"CUDA kullanılabilir mi: "
        f"{torch.cuda.is_available()}"
    )

    article_index_path = create_article_index(
        articles=articles,
        dataset_sha256=dataset_sha256,
    )

    print(
        f"\nMakale-satır eşleştirme dosyası:\n"
        f"{article_index_path}"
    )

    results: List[Dict[str, Any]] = []

    for config in MODEL_CONFIGS:
        result = benchmark_and_save_model(
            config=config,
            articles=articles,
            dataset_sha256=dataset_sha256,
        )

        results.append(result)

    csv_path, json_path = save_summary(
        results
    )

    print("\n" + "=" * 80)
    print("DAY 13 TAMAMLANDI")
    print("=" * 80)

    print(f"\nÖzet CSV:\n{csv_path}")
    print(f"\nÖzet JSON:\n{json_path}")
    print(
        f"\nEmbedding klasörü:\n"
        f"{get_output_directory()}"
    )


if __name__ == "__main__":
    main()