import csv
import gc
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


MODEL_CONFIGS: List[Dict[str, Any]] = [
    {
        "short_name": "MiniLM",
        "model_id": (
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        ),
        "prefix": "",
        "max_seq_length": 128,
        "trust_remote_code": False,
    },
    {
        "short_name": "TR-MTEB",
        "model_id": (
            "trmteb/"
            "turkish-embedding-model-fine-tuned"
        ),
        "prefix": "",
        "max_seq_length": 512,
        "trust_remote_code": False,
    },
    {
        "short_name": "E5-large",
        "model_id": (
            "intfloat/"
            "multilingual-e5-large"
        ),
        "prefix": "query: ",
        "max_seq_length": 512,
        "trust_remote_code": False,
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
    },
]

SAMPLE_SIZE = 200

# Önce 4 denenir. GPU belleği yetmezse otomatik olarak
# 2 ve ardından 1'e düşürülür.
BATCH_SIZE_CANDIDATES = [4, 2, 1]


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


def load_articles() -> List[Dict[str, Any]]:
    """Pilot JSONL veri setini okur."""

    input_path = (
        get_project_root()
        / "data"
        / "processed"
        / "pilot_articles_1000.jsonl"
    )

    articles: List[Dict[str, Any]] = []

    with input_path.open(
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

            abstract = article.get("abstract_tr")

            if (
                isinstance(article, dict)
                and isinstance(abstract, str)
                and abstract.strip()
            ):
                articles.append(article)

    return articles


def select_length_representative_sample(
    articles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    En kısadan en uzuna bütün dağılımı temsil eden
    200 abstract seçer.

    Sadece rastgele seçim yapmadığımız için uzun abstractlar
    da benchmark içinde yer alır.
    """

    sorted_articles = sorted(
        articles,
        key=lambda article: len(
            article["abstract_tr"]
        ),
    )

    if len(sorted_articles) <= SAMPLE_SIZE:
        return sorted_articles

    selected_indices = np.linspace(
        0,
        len(sorted_articles) - 1,
        SAMPLE_SIZE,
        dtype=int,
    )

    return [
        sorted_articles[int(index)]
        for index in selected_indices
    ]


def save_benchmark_sample(
    articles: List[Dict[str, Any]],
) -> Path:
    """Seçilen 200 makaleyi ayrıca kaydeder."""

    output_path = (
        get_project_root()
        / "data"
        / "processed"
        / "benchmark_articles_200.jsonl"
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as output_file:
        for article in articles:
            json.dump(
                article,
                output_file,
                ensure_ascii=False,
            )
            output_file.write("\n")

    return output_path


def synchronize_cuda() -> None:
    """CUDA kullanılıyorsa bekleyen işlemleri tamamlar."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_gpu_memory_mb() -> float:
    """O anda ayrılmış CUDA belleğini MB olarak döndürür."""

    if not torch.cuda.is_available():
        return 0.0

    return (
        torch.cuda.memory_allocated()
        / 1024
        / 1024
    )


def get_peak_gpu_memory_mb() -> float:
    """En yüksek CUDA bellek kullanımını MB olarak döndürür."""

    if not torch.cuda.is_available():
        return 0.0

    return (
        torch.cuda.max_memory_allocated()
        / 1024
        / 1024
    )


def sanitize_filename(value: str) -> str:
    """Model adını güvenli dosya adına dönüştürür."""

    return re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        value,
    ).strip("_").lower()


def encode_with_batch_fallback(
    model: SentenceTransformer,
    texts: List[str],
) -> Tuple[np.ndarray, int, float, float]:
    """
    Metinleri encode eder.

    GPU belleği yetmezse daha küçük batch size ile tekrar dener.
    """

    last_error: Exception = RuntimeError(
        "Embedding üretilemedi."
    )

    for batch_size in BATCH_SIZE_CANDIDATES:
        try:
            print(
                f"Embedding deneniyor: "
                f"batch_size={batch_size}"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Isınma çalışması ölçüm süresine dahil değildir.
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

            start_time = time.perf_counter()

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
                - start_time
            )

            peak_memory_mb = (
                get_peak_gpu_memory_mb()
            )

            return (
                embeddings,
                batch_size,
                encoding_seconds,
                peak_memory_mb,
            )

        except RuntimeError as error:
            last_error = error

            if "out of memory" not in str(error).lower():
                raise

            print(
                f"GPU belleği yetmedi: "
                f"batch_size={batch_size}"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise last_error


def benchmark_model(
    config: Dict[str, Any],
    articles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Tek bir embedding modelini ölçer."""

    print("\n" + "=" * 75)
    print(f"MODEL: {config['short_name']}")
    print("=" * 75)

    print(f"\nModel kimliği: {config['model_id']}")
    print("Model yükleniyor...")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

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

    model_memory_mb = get_gpu_memory_mb()

    texts = [
        config["prefix"] + article["abstract_tr"]
        for article in articles
    ]

    (
        embeddings,
        used_batch_size,
        encoding_seconds,
        peak_memory_mb,
    ) = encode_with_batch_fallback(
        model=model,
        texts=texts,
    )

    output_directory = (
        get_project_root()
        / "research" / "outputs"
        / "day10_embeddings"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedding_path = (
        output_directory
        / (
            sanitize_filename(
                config["short_name"]
            )
            + ".npy"
        )
    )

    np.save(
        embedding_path,
        embeddings,
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
        "device": device,
        "max_seq_length": (
            config["max_seq_length"]
        ),
        "prefix": config["prefix"],
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
            len(articles) / encoding_seconds
        ),
        "model_gpu_memory_mb": model_memory_mb,
        "peak_gpu_memory_mb": peak_memory_mb,
        "vector_file_mb": vector_file_mb,
        "mean_vector_norm": float(
            np.mean(vector_norms)
        ),
        "embedding_path": str(
            embedding_path
        ),
    }

    print(
        f"\nEmbedding şekli : "
        f"{embeddings.shape}"
    )
    print(
        f"Yükleme süresi  : "
        f"{load_seconds:.2f} saniye"
    )
    print(
        f"Encoding süresi : "
        f"{encoding_seconds:.2f} saniye"
    )
    print(
        f"Makale/saniye   : "
        f"{result['articles_per_second']:.2f}"
    )
    print(
        f"GPU tepe bellek : "
        f"{peak_memory_mb:.2f} MB"
    )

    del embeddings
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def save_results(
    results: List[Dict[str, Any]],
) -> None:
    """Benchmark sonuçlarını CSV ve JSON olarak kaydeder."""

    output_directory = (
        get_project_root()
        / "research" / "outputs"
    )

    csv_path = (
        output_directory
        / "day10_embedding_benchmark.csv"
    )

    json_path = (
        output_directory
        / "day10_embedding_benchmark.json"
    )

    fieldnames = list(results[0].keys())

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

    print(f"\nCSV raporu:\n{csv_path}")
    print(f"\nJSON raporu:\n{json_path}")


def main() -> None:
    articles = load_articles()

    sample_articles = (
        select_length_representative_sample(
            articles
        )
    )

    sample_path = save_benchmark_sample(
        sample_articles
    )

    print("=" * 75)
    print("EMBEDDING MODELİ PERFORMANS BENCHMARK'I")
    print("=" * 75)

    print(
        f"\nToplam pilot makale : "
        f"{len(articles)}"
    )
    print(
        f"Benchmark örneklemi : "
        f"{len(sample_articles)}"
    )
    print(f"Çalışma cihazı      : {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"\nÖrneklem dosyası:\n{sample_path}")

    results: List[Dict[str, Any]] = []

    for config in MODEL_CONFIGS:
        result = benchmark_model(
            config=config,
            articles=sample_articles,
        )

        results.append(result)

    save_results(results)

    print("\n" + "=" * 75)
    print("EMBEDDING BENCHMARK'I TAMAMLANDI")
    print("=" * 75)


if __name__ == "__main__":
    main()