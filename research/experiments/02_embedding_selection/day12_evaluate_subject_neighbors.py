import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------
# Deney ayarları
# ---------------------------------------------------------

TOP_K = 5

MODEL_FILES = {
    "MiniLM": "minilm.npy",
    "TR-MTEB": "tr-mteb.npy",
    "E5-large": "e5-large.npy",
    "GTE-multilingual": "gte-multilingual.npy",
}


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


def load_articles() -> List[Dict[str, Any]]:
    """Day 10 benchmark örneklemindeki 200 makaleyi okur."""

    input_path = (
        get_project_root()
        / "data"
        / "processed"
        / "benchmark_articles_200.jsonl"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Benchmark makale dosyası bulunamadı:\n{input_path}"
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

            if isinstance(article, dict):
                articles.append(article)

    return articles


def normalize_embedding_rows(
    embeddings: np.ndarray,
) -> np.ndarray:
    """Embedding satırlarını birim uzunluğa getirir."""

    embeddings = embeddings.astype(
        np.float32,
        copy=False,
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    safe_norms = np.where(
        norms == 0,
        1,
        norms,
    )

    return embeddings / safe_norms


def load_embeddings(
    article_count: int,
) -> Dict[str, np.ndarray]:
    """Day 10'da üretilen bütün embedding dosyalarını yükler."""

    embedding_directory = (
        get_project_root()
        / "research" / "outputs"
        / "day10_embeddings"
    )

    model_embeddings: Dict[str, np.ndarray] = {}

    for model_name, filename in MODEL_FILES.items():
        path = embedding_directory / filename

        if not path.exists():
            raise FileNotFoundError(
                f"{model_name} embedding dosyası bulunamadı:\n{path}"
            )

        embeddings = np.load(path)

        if embeddings.ndim != 2:
            raise ValueError(
                f"{model_name} matrisi iki boyutlu değil: "
                f"{embeddings.shape}"
            )

        if embeddings.shape[0] != article_count:
            raise ValueError(
                f"{model_name} embedding satırı ile "
                f"makale sayısı uyuşmuyor."
            )

        model_embeddings[model_name] = (
            normalize_embedding_rows(embeddings)
        )

        print(
            f"{model_name:18} "
            f"→ şekil={embeddings.shape}"
        )

    return model_embeddings


def parse_subject_item(
    value: Any,
) -> Optional[Dict[str, Any]]:
    """
    Subject kaydını sözlüğe dönüştürür.

    Kayıt doğrudan dict veya JSON stringi olabilir.
    """

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        cleaned_value = value.strip()

        if not cleaned_value:
            return None

        try:
            parsed_value = json.loads(cleaned_value)
        except json.JSONDecodeError:
            return None

        if isinstance(parsed_value, dict):
            return parsed_value

    return None


def get_subject_items(
    article: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Bir makalenin geçerli subject sözlüklerini döndürür."""

    raw_subjects = article.get("subjects")

    if not isinstance(raw_subjects, list):
        return []

    subject_items: List[Dict[str, Any]] = []

    for raw_subject in raw_subjects:
        parsed_subject = parse_subject_item(
            raw_subject
        )

        if parsed_subject is not None:
            subject_items.append(parsed_subject)

    return subject_items


def create_subject_key(
    subject: Dict[str, Any],
) -> Optional[str]:
    """
    Ayrıntılı subjecti karşılaştırmak için kararlı bir anahtar üretir.

    Öncelik:
    1. Subject ID
    2. fullName
    3. name
    """

    subject_id = subject.get("id")

    if subject_id is not None:
        return f"id:{subject_id}"

    full_name = subject.get("fullName")

    if isinstance(full_name, str) and full_name.strip():
        return f"full:{full_name.strip()}"

    name = subject.get("name")

    if isinstance(name, str) and name.strip():
        return f"name:{name.strip()}"

    return None


def create_root_key(
    subject: Dict[str, Any],
) -> Optional[str]:
    """Fen/Sosyal gibi kök subject için anahtar üretir."""

    root_id = subject.get("rootId")

    if root_id is not None:
        return f"root-id:{root_id}"

    root_name = subject.get("rootName")

    if isinstance(root_name, str) and root_name.strip():
        return f"root-name:{root_name.strip()}"

    return None


def get_subject_keys(
    article: Dict[str, Any],
) -> Set[str]:
    """Makalenin ayrıntılı subject anahtarlarını döndürür."""

    keys: Set[str] = set()

    for subject in get_subject_items(article):
        key = create_subject_key(subject)

        if key:
            keys.add(key)

    return keys


def get_root_keys(
    article: Dict[str, Any],
) -> Set[str]:
    """Makalenin Fen/Sosyal kök anahtarlarını döndürür."""

    keys: Set[str] = set()

    for subject in get_subject_items(article):
        key = create_root_key(subject)

        if key:
            keys.add(key)

    return keys


def get_subject_names(
    article: Dict[str, Any],
) -> List[str]:
    """Raporlarda gösterilecek subject isimlerini döndürür."""

    names: List[str] = []

    for subject in get_subject_items(article):
        full_name = subject.get("fullName")
        name = subject.get("name")

        if isinstance(full_name, str) and full_name.strip():
            names.append(full_name.strip())
        elif isinstance(name, str) and name.strip():
            names.append(name.strip())

    return names


def jaccard_similarity(
    first_set: Set[str],
    second_set: Set[str],
) -> float:
    """İki subject kümesinin Jaccard benzerliğini hesaplar."""

    union = first_set | second_set

    if not union:
        return 0.0

    intersection = first_set & second_set

    return len(intersection) / len(union)


def find_neighbors(
    embeddings: np.ndarray,
    anchor_index: int,
    top_k: int,
) -> List[Tuple[int, float]]:
    """Anchor makalenin en yakın top-k komşusunu bulur."""

    anchor_vector = embeddings[anchor_index]

    scores = embeddings @ anchor_vector

    # Makalenin kendisini sonuçlardan çıkar.
    scores[anchor_index] = -np.inf

    ranked_indices = np.argsort(scores)[::-1]

    return [
        (
            int(neighbor_index),
            float(scores[neighbor_index]),
        )
        for neighbor_index in ranked_indices[:top_k]
    ]


def evaluate_model(
    model_name: str,
    embeddings: np.ndarray,
    articles: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Bir modelin komşularını subject etiketleriyle karşılaştırır.

    Subjectler embedding üretiminde kullanılmaz.
    Yalnızca bu değerlendirme aşamasında açılır.
    """

    detail_rows: List[Dict[str, Any]] = []

    evaluable_anchor_count = 0
    total_neighbor_slots = 0
    labeled_neighbor_count = 0

    exact_match_count = 0
    root_match_count = 0

    strict_jaccard_total = 0.0
    labeled_jaccard_total = 0.0

    top1_exact_hit_count = 0
    top1_root_hit_count = 0
    top_k_any_exact_hit_count = 0
    top_k_any_root_hit_count = 0

    for anchor_index, anchor_article in enumerate(articles):
        anchor_subjects = get_subject_keys(
            anchor_article
        )

        anchor_roots = get_root_keys(
            anchor_article
        )

        # Subjecti olmayan anchor değerlendirilemez.
        if not anchor_subjects:
            continue

        evaluable_anchor_count += 1

        neighbors = find_neighbors(
            embeddings=embeddings,
            anchor_index=anchor_index,
            top_k=TOP_K,
        )

        anchor_has_any_exact = False
        anchor_has_any_root = False

        for rank, (
            neighbor_index,
            similarity_score,
        ) in enumerate(
            neighbors,
            start=1,
        ):
            total_neighbor_slots += 1

            neighbor_article = articles[
                neighbor_index
            ]

            neighbor_subjects = get_subject_keys(
                neighbor_article
            )

            neighbor_roots = get_root_keys(
                neighbor_article
            )

            neighbor_is_labeled = bool(
                neighbor_subjects
            )

            if neighbor_is_labeled:
                labeled_neighbor_count += 1

            exact_match = bool(
                anchor_subjects
                & neighbor_subjects
            )

            root_match = bool(
                anchor_roots
                & neighbor_roots
            )

            subject_jaccard = jaccard_similarity(
                anchor_subjects,
                neighbor_subjects,
            )

            strict_jaccard_total += (
                subject_jaccard
            )

            if neighbor_is_labeled:
                labeled_jaccard_total += (
                    subject_jaccard
                )

            if exact_match:
                exact_match_count += 1
                anchor_has_any_exact = True

            if root_match:
                root_match_count += 1
                anchor_has_any_root = True

            if rank == 1 and exact_match:
                top1_exact_hit_count += 1

            if rank == 1 and root_match:
                top1_root_hit_count += 1

            detail_rows.append(
                {
                    "model_name": model_name,
                    "anchor_index": anchor_index,
                    "anchor_article_id": (
                        anchor_article.get(
                            "article_id",
                            "",
                        )
                    ),
                    "anchor_title": (
                        anchor_article.get(
                            "title_tr",
                            "",
                        )
                    ),
                    "anchor_subjects": " | ".join(
                        get_subject_names(
                            anchor_article
                        )
                    ),
                    "rank": rank,
                    "neighbor_index": neighbor_index,
                    "neighbor_article_id": (
                        neighbor_article.get(
                            "article_id",
                            "",
                        )
                    ),
                    "neighbor_title": (
                        neighbor_article.get(
                            "title_tr",
                            "",
                        )
                    ),
                    "neighbor_subjects": " | ".join(
                        get_subject_names(
                            neighbor_article
                        )
                    ),
                    "neighbor_is_labeled": (
                        neighbor_is_labeled
                    ),
                    "similarity_score": (
                        similarity_score
                    ),
                    "exact_subject_match": (
                        exact_match
                    ),
                    "root_subject_match": (
                        root_match
                    ),
                    "subject_jaccard": (
                        subject_jaccard
                    ),
                }
            )

        if anchor_has_any_exact:
            top_k_any_exact_hit_count += 1

        if anchor_has_any_root:
            top_k_any_root_hit_count += 1

    strict_exact_match_rate = (
        exact_match_count
        / total_neighbor_slots
        if total_neighbor_slots
        else 0
    )

    labeled_only_exact_match_rate = (
        exact_match_count
        / labeled_neighbor_count
        if labeled_neighbor_count
        else 0
    )

    strict_root_match_rate = (
        root_match_count
        / total_neighbor_slots
        if total_neighbor_slots
        else 0
    )

    labeled_only_root_match_rate = (
        root_match_count
        / labeled_neighbor_count
        if labeled_neighbor_count
        else 0
    )

    labeled_neighbor_coverage = (
        labeled_neighbor_count
        / total_neighbor_slots
        if total_neighbor_slots
        else 0
    )

    mean_strict_jaccard = (
        strict_jaccard_total
        / total_neighbor_slots
        if total_neighbor_slots
        else 0
    )

    mean_labeled_only_jaccard = (
        labeled_jaccard_total
        / labeled_neighbor_count
        if labeled_neighbor_count
        else 0
    )

    summary = {
        "model_name": model_name,
        "top_k": TOP_K,
        "article_count": len(articles),
        "evaluable_anchor_count": (
            evaluable_anchor_count
        ),
        "total_neighbor_slots": (
            total_neighbor_slots
        ),
        "labeled_neighbor_count": (
            labeled_neighbor_count
        ),
        "labeled_neighbor_coverage": (
            labeled_neighbor_coverage
        ),
        "strict_exact_match_rate": (
            strict_exact_match_rate
        ),
        "labeled_only_exact_match_rate": (
            labeled_only_exact_match_rate
        ),
        "strict_root_match_rate": (
            strict_root_match_rate
        ),
        "labeled_only_root_match_rate": (
            labeled_only_root_match_rate
        ),
        "mean_strict_subject_jaccard": (
            mean_strict_jaccard
        ),
        "mean_labeled_only_subject_jaccard": (
            mean_labeled_only_jaccard
        ),
        "top1_exact_hit_rate": (
            top1_exact_hit_count
            / evaluable_anchor_count
            if evaluable_anchor_count
            else 0
        ),
        "top1_root_hit_rate": (
            top1_root_hit_count
            / evaluable_anchor_count
            if evaluable_anchor_count
            else 0
        ),
        "top_k_any_exact_hit_rate": (
            top_k_any_exact_hit_count
            / evaluable_anchor_count
            if evaluable_anchor_count
            else 0
        ),
        "top_k_any_root_hit_rate": (
            top_k_any_root_hit_count
            / evaluable_anchor_count
            if evaluable_anchor_count
            else 0
        ),
    }

    return summary, detail_rows


def print_summary(
    summaries: List[Dict[str, Any]],
) -> None:
    """Model sonuçlarını terminalde gösterir."""

    print("\n" + "=" * 90)
    print("SUBJECT TABANLI KOMŞULUK KALİTESİ")
    print("=" * 90)

    print(
        "\nNot: Yüzdeler yalnızca subjecti bulunan "
        "anchor ve komşular üzerinden yorumlanmalıdır.\n"
    )

    header = (
        f"{'Model':18}"
        f"{'Anchor':>8}"
        f"{'Etiket kapsamı':>16}"
        f"{'Exact':>11}"
        f"{'Root':>11}"
        f"{'Top1 exact':>13}"
        f"{'Top5 exact':>13}"
    )

    print(header)
    print("-" * len(header))

    for summary in summaries:
        print(
            f"{summary['model_name']:18}"
            f"{summary['evaluable_anchor_count']:>8}"
            f"{summary['labeled_neighbor_coverage'] * 100:>15.2f}%"
            f"{summary['labeled_only_exact_match_rate'] * 100:>10.2f}%"
            f"{summary['labeled_only_root_match_rate'] * 100:>10.2f}%"
            f"{summary['top1_exact_hit_rate'] * 100:>12.2f}%"
            f"{summary['top_k_any_exact_hit_rate'] * 100:>12.2f}%"
        )


def save_results(
    summaries: List[Dict[str, Any]],
    detail_rows: List[Dict[str, Any]],
) -> Tuple[Path, Path, Path]:
    """Özet ve detay sonuçlarını CSV/JSON olarak kaydeder."""

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
        / "day12_subject_neighbor_summary.csv"
    )

    summary_json_path = (
        output_directory
        / "day12_subject_neighbor_summary.json"
    )

    details_csv_path = (
        output_directory
        / "day12_subject_neighbor_details.csv"
    )

    with summary_csv_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                summaries[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(summaries)

    with summary_json_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            summaries,
            json_file,
            ensure_ascii=False,
            indent=2,
        )

    with details_csv_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                detail_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(detail_rows)

    return (
        summary_csv_path,
        summary_json_path,
        details_csv_path,
    )


def create_chart(
    summaries: List[Dict[str, Any]],
) -> Path:
    """Model kalite metriklerini görselleştirir."""

    model_names = [
        summary["model_name"]
        for summary in summaries
    ]

    exact_rates = [
        summary[
            "labeled_only_exact_match_rate"
        ] * 100
        for summary in summaries
    ]

    root_rates = [
        summary[
            "labeled_only_root_match_rate"
        ] * 100
        for summary in summaries
    ]

    coverage_rates = [
        summary[
            "labeled_neighbor_coverage"
        ] * 100
        for summary in summaries
    ]

    x_positions = np.arange(
        len(model_names)
    )

    bar_width = 0.25

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day12_subject_neighbor_quality.png"
    )

    plt.figure(figsize=(12, 7))

    plt.bar(
        x_positions - bar_width,
        exact_rates,
        width=bar_width,
        label="Exact subject eşleşmesi",
    )

    plt.bar(
        x_positions,
        root_rates,
        width=bar_width,
        label="Kök subject eşleşmesi",
    )

    plt.bar(
        x_positions + bar_width,
        coverage_rates,
        width=bar_width,
        label="Etiketli komşu kapsamı",
    )

    plt.xticks(
        x_positions,
        model_names,
    )

    plt.ylabel("Oran (%)")

    plt.title(
        "Embedding Modellerinin Subject Tabanlı "
        "Komşuluk Kalitesi"
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=160,
    )
    plt.close()

    return output_path


def main() -> None:
    print("=" * 90)
    print("SUBJECT TABANLI SEMANTIC KOMŞULUK DEĞERLENDİRMESİ")
    print("=" * 90)

    articles = load_articles()

    labeled_article_count = sum(
        bool(get_subject_keys(article))
        for article in articles
    )

    print(f"\nToplam makale          : {len(articles)}")
    print(
        f"Subject bulunan makale: "
        f"{labeled_article_count}"
    )

    print("\nEmbedding dosyaları:")

    model_embeddings = load_embeddings(
        article_count=len(articles)
    )

    summaries: List[Dict[str, Any]] = []
    all_detail_rows: List[Dict[str, Any]] = []

    for model_name, embeddings in (
        model_embeddings.items()
    ):
        summary, detail_rows = evaluate_model(
            model_name=model_name,
            embeddings=embeddings,
            articles=articles,
        )

        summaries.append(summary)
        all_detail_rows.extend(
            detail_rows
        )

    print_summary(summaries)

    (
        summary_csv_path,
        summary_json_path,
        details_csv_path,
    ) = save_results(
        summaries=summaries,
        detail_rows=all_detail_rows,
    )

    chart_path = create_chart(
        summaries
    )

    print("\n" + "=" * 90)
    print("DOSYALAR OLUŞTURULDU")
    print("=" * 90)

    print(f"\nÖzet CSV:\n{summary_csv_path}")
    print(f"\nÖzet JSON:\n{summary_json_path}")
    print(f"\nDetay CSV:\n{details_csv_path}")
    print(f"\nGörsel:\n{chart_path}")


if __name__ == "__main__":
    main()