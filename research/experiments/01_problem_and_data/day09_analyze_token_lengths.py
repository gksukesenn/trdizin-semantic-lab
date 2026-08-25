import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

# Grafik arayüz açılmadan PNG dosyasına kaydedilecek.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer


# ---------------------------------------------------------
# 1. İnceleyeceğimiz embedding modeli tokenizer'ları
# ---------------------------------------------------------
#
# Bu aşamada embedding modellerinin ağırlıklarını yüklemiyoruz.
# Yalnızca her modelin tokenizer'ını indiriyoruz.
#
# max_tokens:
# Modelin işleyeceği maksimum token sayısı.
#
# prefix:
# Abstractı modele vermeden önce eklenecek metin.
# E5, clustering gibi retrieval dışı görevlerde de
# "query: " önekini öneriyor.
#

MODEL_CONFIGS: List[Dict[str, Any]] = [
    {
        "short_name": "MiniLM",
        "model_id": (
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        ),
        "max_tokens": 128,
        "prefix": "",
        "trust_remote_code": False,
    },
    {
        "short_name": "TR-MTEB",
        "model_id": (
            "trmteb/"
            "turkish-embedding-model-fine-tuned"
        ),
        "max_tokens": 512,
        "prefix": "",
        "trust_remote_code": False,
    },
    {
        "short_name": "E5-large",
        "model_id": (
            "intfloat/"
            "multilingual-e5-large"
        ),
        "max_tokens": 512,
        "prefix": "query: ",
        "trust_remote_code": False,
    },
    {
        "short_name": "GTE-multilingual",
        "model_id": (
            "Alibaba-NLP/"
            "gte-multilingual-base"
        ),
        "max_tokens": 8192,
        "prefix": "",
        "trust_remote_code": True,
    },
]


BATCH_SIZE = 64


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


def load_pilot_articles() -> List[Dict[str, Any]]:
    """
    Day 08 aşamasında oluşturulan JSONL
    pilot veri setini okur.
    """

    input_path = (
        get_project_root()
        / "data"
        / "processed"
        / "pilot_articles_1000.jsonl"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Pilot veri dosyası bulunamadı:\n{input_path}"
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

            if not isinstance(article, dict):
                continue

            abstract = article.get("abstract_tr")

            if not isinstance(abstract, str):
                continue

            if not abstract.strip():
                continue

            articles.append(article)

    return articles


def tokenize_in_batches(
    tokenizer: Any,
    texts: List[str],
) -> List[int]:
    """
    Metinleri parça parça tokenize eder.

    truncation=False:
    Metinleri kesmeden gerçek token sayılarını ölçmemizi sağlar.

    Henüz embedding oluşturulmaz.
    """

    token_lengths: List[int] = []

    for batch_start in range(
        0,
        len(texts),
        BATCH_SIZE,
    ):
        batch_end = batch_start + BATCH_SIZE
        text_batch = texts[batch_start:batch_end]

        encoded_batch = tokenizer(
            text_batch,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
        )

        lengths = encoded_batch.get("length")

        if lengths is not None:
            token_lengths.extend(
                int(length)
                for length in lengths
            )
        else:
            # Bazı tokenizer'lar length alanı döndürmezse
            # input_ids uzunluklarını kullanırız.
            input_ids = encoded_batch["input_ids"]

            token_lengths.extend(
                len(token_ids)
                for token_ids in input_ids
            )

        print(
            f"  İşlenen: "
            f"{min(batch_end, len(texts))}/{len(texts)}"
        )

    return token_lengths


def calculate_model_summary(
    config: Dict[str, Any],
    token_lengths: List[int],
) -> Dict[str, Any]:
    """Bir model için token ve kesilme istatistiklerini hesaplar."""

    max_tokens = int(config["max_tokens"])

    truncated_lengths = [
        token_count
        for token_count in token_lengths
        if token_count > max_tokens
    ]

    truncated_article_count = len(
        truncated_lengths
    )

    total_article_count = len(token_lengths)

    truncation_percentage = (
        truncated_article_count
        / total_article_count
        * 100
        if total_article_count
        else 0
    )

    lost_token_count = sum(
        token_count - max_tokens
        for token_count in truncated_lengths
    )

    return {
        "short_name": config["short_name"],
        "model_id": config["model_id"],
        "model_limit": max_tokens,
        "prefix": config["prefix"],
        "article_count": total_article_count,
        "minimum_tokens": int(
            min(token_lengths)
        ),
        "maximum_tokens": int(
            max(token_lengths)
        ),
        "mean_tokens": float(
            np.mean(token_lengths)
        ),
        "median_tokens": float(
            np.median(token_lengths)
        ),
        "p90_tokens": float(
            np.percentile(token_lengths, 90)
        ),
        "p95_tokens": float(
            np.percentile(token_lengths, 95)
        ),
        "p99_tokens": float(
            np.percentile(token_lengths, 99)
        ),
        "truncated_article_count": (
            truncated_article_count
        ),
        "truncation_percentage": (
            truncation_percentage
        ),
        "total_lost_token_count": (
            lost_token_count
        ),
        "over_128_count": sum(
            token_count > 128
            for token_count in token_lengths
        ),
        "over_512_count": sum(
            token_count > 512
            for token_count in token_lengths
        ),
        "over_8192_count": sum(
            token_count > 8192
            for token_count in token_lengths
        ),
    }


def analyze_model_tokenizer(
    config: Dict[str, Any],
    articles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Bir tokenizer'ı yükler ve bütün abstractları ölçer."""

    print("\n" + "=" * 75)
    print(f"MODEL: {config['short_name']}")
    print("=" * 75)

    print(f"\nModel kimliği : {config['model_id']}")
    print(f"Model sınırı  : {config['max_tokens']}")
    print(f"Metin öneki   : {config['prefix']!r}")
    print("\nTokenizer yükleniyor...")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        trust_remote_code=config[
            "trust_remote_code"
        ],
    )

    texts = [
        config["prefix"] + article["abstract_tr"]
        for article in articles
    ]

    token_lengths = tokenize_in_batches(
        tokenizer=tokenizer,
        texts=texts,
    )

    summary = calculate_model_summary(
        config=config,
        token_lengths=token_lengths,
    )

    detail_rows: List[Dict[str, Any]] = []

    for article, token_count in zip(
        articles,
        token_lengths,
    ):
        detail_rows.append(
            {
                "article_id": article["article_id"],
                "publication_year": article[
                    "publication_year"
                ],
                "model_name": config["short_name"],
                "model_id": config["model_id"],
                "token_count": token_count,
                "model_limit": config["max_tokens"],
                "would_truncate": (
                    token_count
                    > config["max_tokens"]
                ),
                "tokens_over_limit": max(
                    token_count
                    - config["max_tokens"],
                    0,
                ),
                "abstract_character_count": len(
                    article["abstract_tr"]
                ),
                "title_tr": article.get(
                    "title_tr",
                    "",
                ),
            }
        )

    summary["detail_rows"] = detail_rows

    return summary


def print_summary(
    model_summaries: List[Dict[str, Any]],
) -> None:
    """Sonuçları terminalde okunabilir biçimde gösterir."""

    print("\n" + "=" * 75)
    print("TOKEN SINIRI KARŞILAŞTIRMASI")
    print("=" * 75)

    for summary in model_summaries:
        print("\n" + "-" * 75)
        print(f"Model       : {summary['short_name']}")
        print(f"Sınır       : {summary['model_limit']}")
        print(
            f"Minimum     : "
            f"{summary['minimum_tokens']}"
        )
        print(
            f"Maksimum    : "
            f"{summary['maximum_tokens']}"
        )
        print(
            f"Ortalama    : "
            f"{summary['mean_tokens']:.2f}"
        )
        print(
            f"Medyan      : "
            f"{summary['median_tokens']:.2f}"
        )
        print(
            f"Yüzde 90    : "
            f"{summary['p90_tokens']:.2f}"
        )
        print(
            f"Yüzde 95    : "
            f"{summary['p95_tokens']:.2f}"
        )
        print(
            f"Kesilecek   : "
            f"{summary['truncated_article_count']}"
            f"/{summary['article_count']}"
        )
        print(
            f"Kesilme oranı: "
            f"%{summary['truncation_percentage']:.2f}"
        )


def save_summary_files(
    model_summaries: List[Dict[str, Any]],
) -> None:
    """Özet ve makale bazlı sonuçları CSV/JSON olarak kaydeder."""

    output_directory = (
        get_project_root()
        / "research" / "outputs"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_csv_path = (
        output_directory
        / "day09_token_length_summary.csv"
    )

    summary_json_path = (
        output_directory
        / "day09_token_length_summary.json"
    )

    details_csv_path = (
        output_directory
        / "day09_token_length_details.csv"
    )

    summary_fieldnames = [
        "short_name",
        "model_id",
        "model_limit",
        "prefix",
        "article_count",
        "minimum_tokens",
        "maximum_tokens",
        "mean_tokens",
        "median_tokens",
        "p90_tokens",
        "p95_tokens",
        "p99_tokens",
        "truncated_article_count",
        "truncation_percentage",
        "total_lost_token_count",
        "over_128_count",
        "over_512_count",
        "over_8192_count",
    ]

    with summary_csv_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=summary_fieldnames,
        )

        writer.writeheader()

        for summary in model_summaries:
            writer.writerow(
                {
                    key: summary[key]
                    for key in summary_fieldnames
                }
            )

    json_summaries = [
        {
            key: summary[key]
            for key in summary_fieldnames
        }
        for summary in model_summaries
    ]

    with summary_json_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            json_summaries,
            json_file,
            ensure_ascii=False,
            indent=2,
        )

    detail_fieldnames = [
        "article_id",
        "publication_year",
        "model_name",
        "model_id",
        "token_count",
        "model_limit",
        "would_truncate",
        "tokens_over_limit",
        "abstract_character_count",
        "title_tr",
    ]

    with details_csv_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=detail_fieldnames,
        )

        writer.writeheader()

        for summary in model_summaries:
            writer.writerows(
                summary["detail_rows"]
            )

    print("\nDosyalar:")
    print(summary_csv_path)
    print(summary_json_path)
    print(details_csv_path)


def create_truncation_chart(
    model_summaries: List[Dict[str, Any]],
) -> Path:
    """Modellerin kesilme oranlarını çubuk grafikle gösterir."""

    model_names = [
        summary["short_name"]
        for summary in model_summaries
    ]

    percentages = [
        summary["truncation_percentage"]
        for summary in model_summaries
    ]

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day09_truncation_rates.png"
    )

    plt.figure(figsize=(11, 6))
    bars = plt.bar(
        model_names,
        percentages,
    )

    for bar, percentage in zip(
        bars,
        percentages,
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"%{percentage:.1f}",
            ha="center",
            va="bottom",
        )

    plt.title(
        "Model Token Sınırı Nedeniyle Kesilecek Abstract Oranı"
    )
    plt.xlabel("Embedding modeli")
    plt.ylabel("Kesilecek makale oranı (%)")
    plt.ylim(
        0,
        max(percentages + [5]) * 1.15,
    )
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=160,
    )
    plt.close()

    return output_path


def main() -> None:
    articles = load_pilot_articles()

    print("=" * 75)
    print("PİLOT ABSTRACT TOKEN ANALİZİ")
    print("=" * 75)
    print(f"\nOkunan makale sayısı: {len(articles)}")

    model_summaries: List[Dict[str, Any]] = []

    for config in MODEL_CONFIGS:
        summary = analyze_model_tokenizer(
            config=config,
            articles=articles,
        )

        model_summaries.append(summary)

    print_summary(model_summaries)
    save_summary_files(model_summaries)

    chart_path = create_truncation_chart(
        model_summaries
    )

    print(f"\nGörsel:\n{chart_path}")

    print("\n" + "=" * 75)
    print("TOKEN ANALİZİ TAMAMLANDI")
    print("=" * 75)


if __name__ == "__main__":
    main()