import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


HDBSCAN_CONFIG_ID = "H16"
TOP_NOISE_SUBJECT_COUNT = 25


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


# =========================================================
# 1. VERİYİ OKUMA
# =========================================================


def load_articles() -> List[Dict[str, Any]]:
    """1.000 makalelik pilot veri setini okur."""

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

            if isinstance(article, dict):
                articles.append(article)

    if len(articles) != 1000:
        raise ValueError(
            "1.000 makale bekleniyordu, "
            f"bulunan: {len(articles)}"
        )

    return articles


def load_kmeans_assignments() -> List[Dict[str, Any]]:
    """Day 15 KMeans k=30 atamalarını okur."""

    input_path = (
        get_project_root()
        / "research" / "outputs"
        / "day15_tr_mteb_k30_assignments.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"KMeans atama dosyası bulunamadı:\n{input_path}"
        )

    rows: List[Dict[str, Any]] = []

    with input_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = {
            "row_index",
            "article_id",
            "cluster_id",
        }

        available_columns = set(
            reader.fieldnames or []
        )

        missing_columns = (
            required_columns - available_columns
        )

        if missing_columns:
            raise ValueError(
                "KMeans CSV dosyasında eksik sütunlar var: "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            rows.append(
                {
                    "row_index": int(row["row_index"]),
                    "article_id": row["article_id"],
                    "label": int(row["cluster_id"]),
                }
            )

    rows.sort(
        key=lambda row: row["row_index"]
    )

    if len(rows) != 1000:
        raise ValueError(
            "1.000 KMeans ataması bekleniyordu, "
            f"bulunan: {len(rows)}"
        )

    return rows


def load_h16_assignments() -> List[Dict[str, Any]]:
    """Day 17 dosyasından H16 HDBSCAN atamalarını okur."""

    input_path = (
        get_project_root()
        / "research" / "outputs"
        / "day17_hdbscan_all_assignments.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"HDBSCAN atama dosyası bulunamadı:\n{input_path}"
        )

    label_column = (
        f"{HDBSCAN_CONFIG_ID}_label"
    )

    probability_column = (
        f"{HDBSCAN_CONFIG_ID}_probability"
    )

    rows: List[Dict[str, Any]] = []

    with input_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = {
            "row_index",
            "article_id",
            label_column,
            probability_column,
        }

        available_columns = set(
            reader.fieldnames or []
        )

        missing_columns = (
            required_columns - available_columns
        )

        if missing_columns:
            raise ValueError(
                "HDBSCAN CSV dosyasında eksik sütunlar var: "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            rows.append(
                {
                    "row_index": int(row["row_index"]),
                    "article_id": row["article_id"],
                    "label": int(row[label_column]),
                    "probability": float(
                        row[probability_column]
                    ),
                }
            )

    rows.sort(
        key=lambda row: row["row_index"]
    )

    if len(rows) != 1000:
        raise ValueError(
            "1.000 HDBSCAN ataması bekleniyordu, "
            f"bulunan: {len(rows)}"
        )

    return rows


def validate_alignment(
    articles: List[Dict[str, Any]],
    kmeans_rows: List[Dict[str, Any]],
    hdbscan_rows: List[Dict[str, Any]],
) -> None:
    """Üç kaynağın aynı makale sırasında olduğunu doğrular."""

    for row_index in range(len(articles)):
        article_id = str(
            articles[row_index].get(
                "article_id",
                "",
            )
        )

        if kmeans_rows[row_index]["row_index"] != row_index:
            raise ValueError(
                f"KMeans row_index uyuşmazlığı: {row_index}"
            )

        if hdbscan_rows[row_index]["row_index"] != row_index:
            raise ValueError(
                f"HDBSCAN row_index uyuşmazlığı: {row_index}"
            )

        if kmeans_rows[row_index]["article_id"] != article_id:
            raise ValueError(
                f"KMeans article_id uyuşmazlığı: {row_index}"
            )

        if hdbscan_rows[row_index]["article_id"] != article_id:
            raise ValueError(
                f"HDBSCAN article_id uyuşmazlığı: {row_index}"
            )


# =========================================================
# 2. SUBJECT ALANLARINI HAZIRLAMA
# =========================================================


def parse_subject_item(
    value: Any,
) -> Optional[Dict[str, Any]]:
    """Subject kaydını sözlüğe dönüştürür."""

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        cleaned_value = value.strip()

        if not cleaned_value:
            return None

        try:
            parsed_value = json.loads(
                cleaned_value
            )
        except json.JSONDecodeError:
            return None

        if isinstance(parsed_value, dict):
            return parsed_value

    return None


def create_subject_key(
    subject: Dict[str, Any],
) -> Optional[str]:
    """Ayrıntılı subject için kararlı anahtar üretir."""

    subject_id = subject.get("id")

    if subject_id is not None:
        return f"id:{subject_id}"

    full_name = subject.get("fullName")

    if (
        isinstance(full_name, str)
        and full_name.strip()
    ):
        return (
            "full:"
            + full_name.strip()
        )

    name = subject.get("name")

    if (
        isinstance(name, str)
        and name.strip()
    ):
        return (
            "name:"
            + name.strip()
        )

    return None


def create_root_key(
    subject: Dict[str, Any],
) -> Optional[str]:
    """Fen/Sosyal gibi kök alan anahtarı üretir."""

    root_id = subject.get("rootId")

    if root_id is not None:
        return f"root-id:{root_id}"

    root_name = subject.get("rootName")

    if (
        isinstance(root_name, str)
        and root_name.strip()
    ):
        return (
            "root-name:"
            + root_name.strip()
        )

    return None


def get_subject_display_name(
    subject: Dict[str, Any],
) -> Optional[str]:
    """Subjectin raporda kullanılacak adını döndürür."""

    full_name = subject.get("fullName")

    if (
        isinstance(full_name, str)
        and full_name.strip()
    ):
        return full_name.strip()

    name = subject.get("name")

    if (
        isinstance(name, str)
        and name.strip()
    ):
        return name.strip()

    return None


def get_root_display_name(
    subject: Dict[str, Any],
) -> Optional[str]:
    """Kök subjectin raporda kullanılacak adını döndürür."""

    root_name = subject.get("rootName")

    if (
        isinstance(root_name, str)
        and root_name.strip()
    ):
        return root_name.strip()

    return None


def build_subject_information(
    articles: List[Dict[str, Any]],
) -> Tuple[
    List[Set[str]],
    List[Set[str]],
    Dict[str, str],
    Dict[str, str],
]:
    """
    Her makale için subject ve root kümelerini oluşturur.

    Aynı subject tek makalede tekrar geçse bile
    yalnızca bir kez sayılır.
    """

    article_subject_sets: List[
        Set[str]
    ] = []

    article_root_sets: List[
        Set[str]
    ] = []

    subject_display_names: Dict[
        str,
        str,
    ] = {}

    root_display_names: Dict[
        str,
        str,
    ] = {}

    for article in articles:
        subject_keys: Set[str] = set()
        root_keys: Set[str] = set()

        raw_subjects = article.get(
            "subjects"
        )

        if not isinstance(
            raw_subjects,
            list,
        ):
            raw_subjects = []

        for raw_subject in raw_subjects:
            subject = parse_subject_item(
                raw_subject
            )

            if subject is None:
                continue

            subject_key = create_subject_key(
                subject
            )

            root_key = create_root_key(
                subject
            )

            subject_name = (
                get_subject_display_name(
                    subject
                )
            )

            root_name = (
                get_root_display_name(
                    subject
                )
            )

            if subject_key:
                subject_keys.add(
                    subject_key
                )

                if subject_name:
                    subject_display_names.setdefault(
                        subject_key,
                        subject_name,
                    )

            if root_key:
                root_keys.add(
                    root_key
                )

                if root_name:
                    root_display_names.setdefault(
                        root_key,
                        root_name,
                    )

        article_subject_sets.append(
            subject_keys
        )

        article_root_sets.append(
            root_keys
        )

    return (
        article_subject_sets,
        article_root_sets,
        subject_display_names,
        root_display_names,
    )


# =========================================================
# 3. CLUSTER KALİTE METRİKLERİ
# =========================================================


def calculate_pair_metrics(
    subject_sets: List[Set[str]],
) -> Dict[str, Any]:
    """
    Cluster içindeki etiketli makale çiftlerinin
    Jaccard ve ortak-subject oranını hesaplar.
    """

    pair_count = 0
    jaccard_total = 0.0
    overlap_count = 0

    for first_index in range(
        len(subject_sets)
    ):
        for second_index in range(
            first_index + 1,
            len(subject_sets),
        ):
            first_set = subject_sets[
                first_index
            ]

            second_set = subject_sets[
                second_index
            ]

            union = first_set | second_set
            intersection = (
                first_set & second_set
            )

            if not union:
                continue

            pair_count += 1

            jaccard_total += (
                len(intersection)
                / len(union)
            )

            if intersection:
                overlap_count += 1

    return {
        "pair_count": pair_count,
        "jaccard_total": (
            jaccard_total
        ),
        "overlap_count": (
            overlap_count
        ),
        "mean_jaccard": (
            jaccard_total / pair_count
            if pair_count
            else None
        ),
        "overlap_rate": (
            overlap_count / pair_count
            if pair_count
            else None
        ),
    }


def evaluate_partition(
    evaluation_name: str,
    labels: np.ndarray,
    active_mask: np.ndarray,
    article_subject_sets: List[Set[str]],
    article_root_sets: List[Set[str]],
    subject_display_names: Dict[str, str],
    root_display_names: Dict[str, str],
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    """
    Bir clustering çözümünün subject kalitesini ölçer.

    active_mask:
    - KMeans all için bütün makaleler True
    - H16 için yalnızca label >= 0
    - Ortak alt küme için H16'nın clusterladığı makaleler
    """

    active_indices = np.where(
        active_mask
    )[0]

    active_cluster_ids = sorted(
        {
            int(labels[index])
            for index in active_indices
            if int(labels[index]) >= 0
        }
    )

    detail_rows: List[
        Dict[str, Any]
    ] = []

    total_assigned_count = int(
        len(active_indices)
    )

    total_labeled_count = 0
    total_root_labeled_count = 0

    dominant_subject_count_total = 0
    dominant_root_count_total = 0

    subject_purities: List[
        float
    ] = []

    root_purities: List[
        float
    ] = []

    total_pair_count = 0
    total_jaccard = 0.0
    total_overlap_count = 0

    for cluster_id in active_cluster_ids:
        cluster_indices = [
            int(index)
            for index in active_indices
            if int(labels[index])
            == cluster_id
        ]

        subject_labeled_indices = [
            index
            for index in cluster_indices
            if article_subject_sets[
                index
            ]
        ]

        root_labeled_indices = [
            index
            for index in cluster_indices
            if article_root_sets[
                index
            ]
        ]

        subject_counter = Counter()

        for article_index in (
            subject_labeled_indices
        ):
            for subject_key in (
                article_subject_sets[
                    article_index
                ]
            ):
                subject_counter[
                    subject_key
                ] += 1

        root_counter = Counter()

        for article_index in (
            root_labeled_indices
        ):
            for root_key in (
                article_root_sets[
                    article_index
                ]
            ):
                root_counter[
                    root_key
                ] += 1

        if subject_counter:
            (
                dominant_subject_key,
                dominant_subject_count,
            ) = subject_counter.most_common(
                1
            )[0]

            subject_purity = (
                dominant_subject_count
                / len(
                    subject_labeled_indices
                )
            )

            dominant_subject_name = (
                subject_display_names.get(
                    dominant_subject_key,
                    dominant_subject_key,
                )
            )
        else:
            dominant_subject_key = ""
            dominant_subject_name = ""
            dominant_subject_count = 0
            subject_purity = None

        if root_counter:
            (
                dominant_root_key,
                dominant_root_count,
            ) = root_counter.most_common(
                1
            )[0]

            root_purity = (
                dominant_root_count
                / len(
                    root_labeled_indices
                )
            )

            dominant_root_name = (
                root_display_names.get(
                    dominant_root_key,
                    dominant_root_key,
                )
            )
        else:
            dominant_root_key = ""
            dominant_root_name = ""
            dominant_root_count = 0
            root_purity = None

        pair_metrics = calculate_pair_metrics(
            [
                article_subject_sets[index]
                for index
                in subject_labeled_indices
            ]
        )

        total_labeled_count += len(
            subject_labeled_indices
        )

        total_root_labeled_count += len(
            root_labeled_indices
        )

        dominant_subject_count_total += (
            dominant_subject_count
        )

        dominant_root_count_total += (
            dominant_root_count
        )

        if subject_purity is not None:
            subject_purities.append(
                subject_purity
            )

        if root_purity is not None:
            root_purities.append(
                root_purity
            )

        total_pair_count += (
            pair_metrics["pair_count"]
        )

        total_jaccard += (
            pair_metrics["jaccard_total"]
        )

        total_overlap_count += (
            pair_metrics["overlap_count"]
        )

        detail_rows.append(
            {
                "evaluation_name": (
                    evaluation_name
                ),
                "cluster_id": cluster_id,
                "cluster_size": len(
                    cluster_indices
                ),
                "subject_labeled_count": len(
                    subject_labeled_indices
                ),
                "subject_labeled_rate": (
                    len(
                        subject_labeled_indices
                    )
                    / len(cluster_indices)
                    if cluster_indices
                    else 0.0
                ),
                "dominant_subject": (
                    dominant_subject_name
                ),
                "dominant_subject_count": (
                    dominant_subject_count
                ),
                "subject_purity": (
                    subject_purity
                ),
                "root_labeled_count": len(
                    root_labeled_indices
                ),
                "dominant_root": (
                    dominant_root_name
                ),
                "dominant_root_count": (
                    dominant_root_count
                ),
                "root_purity": (
                    root_purity
                ),
                "subject_pair_count": (
                    pair_metrics[
                        "pair_count"
                    ]
                ),
                "mean_subject_jaccard": (
                    pair_metrics[
                        "mean_jaccard"
                    ]
                ),
                "subject_overlap_rate": (
                    pair_metrics[
                        "overlap_rate"
                    ]
                ),
            }
        )

    summary = {
        "evaluation_name": (
            evaluation_name
        ),
        "cluster_count": len(
            active_cluster_ids
        ),
        "assigned_article_count": (
            total_assigned_count
        ),
        "assigned_coverage": (
            total_assigned_count
            / len(labels)
        ),
        "subject_labeled_article_count": (
            total_labeled_count
        ),
        "subject_labeled_rate": (
            total_labeled_count
            / total_assigned_count
            if total_assigned_count
            else 0.0
        ),
        "weighted_subject_purity": (
            dominant_subject_count_total
            / total_labeled_count
            if total_labeled_count
            else 0.0
        ),
        "macro_subject_purity": (
            float(
                np.mean(subject_purities)
            )
            if subject_purities
            else 0.0
        ),
        "weighted_root_purity": (
            dominant_root_count_total
            / total_root_labeled_count
            if total_root_labeled_count
            else 0.0
        ),
        "macro_root_purity": (
            float(
                np.mean(root_purities)
            )
            if root_purities
            else 0.0
        ),
        "subject_pair_count": (
            total_pair_count
        ),
        "weighted_mean_subject_jaccard": (
            total_jaccard
            / total_pair_count
            if total_pair_count
            else 0.0
        ),
        "subject_pair_overlap_rate": (
            total_overlap_count
            / total_pair_count
            if total_pair_count
            else 0.0
        ),
    }

    return summary, detail_rows


# =========================================================
# 4. H16 NOISE ANALİZİ
# =========================================================


def analyze_noise_subjects(
    hdbscan_labels: np.ndarray,
    article_subject_sets: List[Set[str]],
    article_root_sets: List[Set[str]],
    subject_display_names: Dict[str, str],
    root_display_names: Dict[str, str],
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    """H16 noise makalelerinin subject dağılımını çıkarır."""

    noise_indices = np.where(
        hdbscan_labels == -1
    )[0]

    subject_counter = Counter()
    root_counter = Counter()

    labeled_noise_count = 0
    root_labeled_noise_count = 0

    for article_index in noise_indices:
        subject_set = (
            article_subject_sets[
                int(article_index)
            ]
        )

        root_set = (
            article_root_sets[
                int(article_index)
            ]
        )

        if subject_set:
            labeled_noise_count += 1

        if root_set:
            root_labeled_noise_count += 1

        for subject_key in subject_set:
            subject_counter[
                subject_key
            ] += 1

        for root_key in root_set:
            root_counter[
                root_key
            ] += 1

    distribution_rows: List[
        Dict[str, Any]
    ] = []

    for rank, (
        subject_key,
        count,
    ) in enumerate(
        subject_counter.most_common(
            TOP_NOISE_SUBJECT_COUNT
        ),
        start=1,
    ):
        distribution_rows.append(
            {
                "rank": rank,
                "subject": (
                    subject_display_names.get(
                        subject_key,
                        subject_key,
                    )
                ),
                "article_count": count,
                "share_of_labeled_noise": (
                    count
                    / labeled_noise_count
                    if labeled_noise_count
                    else 0.0
                ),
            }
        )

    root_distribution = [
        {
            "root": (
                root_display_names.get(
                    root_key,
                    root_key,
                )
            ),
            "article_count": count,
            "share_of_root_labeled_noise": (
                count
                / root_labeled_noise_count
                if root_labeled_noise_count
                else 0.0
            ),
        }
        for root_key, count
        in root_counter.most_common()
    ]

    summary = {
        "noise_article_count": int(
            len(noise_indices)
        ),
        "subject_labeled_noise_count": (
            labeled_noise_count
        ),
        "subject_labeled_noise_rate": (
            labeled_noise_count
            / len(noise_indices)
            if len(noise_indices)
            else 0.0
        ),
        "root_labeled_noise_count": (
            root_labeled_noise_count
        ),
        "root_distribution": (
            root_distribution
        ),
    }

    return summary, distribution_rows


# =========================================================
# 5. DOSYALARI KAYDETME
# =========================================================


def save_summary_csv(
    summaries: List[Dict[str, Any]],
) -> Path:
    """Karşılaştırma özetini CSV olarak kaydeder."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day20_cluster_subject_quality_summary.csv"
    )

    with output_path.open(
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
        writer.writerows(
            summaries
        )

    return output_path


def save_detail_csv(
    detail_rows: List[Dict[str, Any]],
) -> Path:
    """Cluster bazlı detayları CSV olarak kaydeder."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day20_cluster_subject_quality_details.csv"
    )

    with output_path.open(
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
        writer.writerows(
            detail_rows
        )

    return output_path


def save_noise_csv(
    noise_rows: List[Dict[str, Any]],
) -> Path:
    """Noise subject dağılımını CSV olarak kaydeder."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day20_h16_noise_subject_distribution.csv"
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                noise_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            noise_rows
        )

    return output_path


def create_quality_chart(
    summaries: List[Dict[str, Any]],
) -> Path:
    """Ana kalite metriklerini karşılaştıran grafik üretir."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day20_cluster_subject_quality.png"
    )

    evaluation_names = [
        summary["evaluation_name"]
        for summary in summaries
    ]

    subject_purity = [
        summary[
            "weighted_subject_purity"
        ] * 100
        for summary in summaries
    ]

    root_purity = [
        summary[
            "weighted_root_purity"
        ] * 100
        for summary in summaries
    ]

    overlap_rate = [
        summary[
            "subject_pair_overlap_rate"
        ] * 100
        for summary in summaries
    ]

    x_positions = np.arange(
        len(evaluation_names)
    )

    bar_width = 0.25

    plt.figure(
        figsize=(14, 8)
    )

    plt.bar(
        x_positions - bar_width,
        subject_purity,
        width=bar_width,
        label="Ağırlıklı subject saflığı",
    )

    plt.bar(
        x_positions,
        root_purity,
        width=bar_width,
        label="Ağırlıklı kök saflığı",
    )

    plt.bar(
        x_positions + bar_width,
        overlap_rate,
        width=bar_width,
        label="Ortak subjectli çift oranı",
    )

    plt.xticks(
        x_positions,
        evaluation_names,
        rotation=12,
        ha="right",
    )

    plt.ylabel("Oran (%)")

    plt.title(
        "KMeans ve H16: Subject Tabanlı Cluster Kalitesi"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=170,
    )

    plt.close()

    return output_path


def save_markdown_report(
    summaries: List[Dict[str, Any]],
    noise_summary: Dict[str, Any],
    noise_rows: List[Dict[str, Any]],
) -> Path:
    """Sonuçların okunabilir Markdown raporunu oluşturur."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day20_cluster_subject_quality_report.md"
    )

    summary_by_name = {
        summary["evaluation_name"]:
        summary
        for summary in summaries
    }

    h16_summary = summary_by_name[
        "H16_clustered_articles"
    ]

    kmeans_common_summary = (
        summary_by_name[
            "KMeans_same_H16_subset"
        ]
    )

    subject_purity_difference = (
        h16_summary[
            "weighted_subject_purity"
        ]
        - kmeans_common_summary[
            "weighted_subject_purity"
        ]
    )

    overlap_difference = (
        h16_summary[
            "subject_pair_overlap_rate"
        ]
        - kmeans_common_summary[
            "subject_pair_overlap_rate"
        ]
    )

    lines: List[str] = [
        "# KMeans ve H16 Subject Kalitesi Karşılaştırması",
        "",
        (
            "Subjectler clustering girdisi olarak kullanılmamıştır. "
            "Yalnızca sonuçları sonradan değerlendirmek için "
            "kullanılmıştır."
        ),
        "",
        (
            "TR Dizin subjectleri çok etiketlidir ve kusursuz "
            "ground truth olarak kabul edilmemelidir. "
            "Bu ölçümler bir değerlendirme göstergesidir."
        ),
        "",
        "## Özet sonuçlar",
        "",
        (
            "| Değerlendirme | Cluster | Atanan makale | "
            "Kapsam | Subject saflığı | Kök saflığı | "
            "Ortak subjectli çift | Ortalama Jaccard |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for summary in summaries:
        lines.append(
            f"| {summary['evaluation_name']} "
            f"| {summary['cluster_count']} "
            f"| {summary['assigned_article_count']} "
            f"| %{summary['assigned_coverage'] * 100:.2f} "
            f"| %{summary['weighted_subject_purity'] * 100:.2f} "
            f"| %{summary['weighted_root_purity'] * 100:.2f} "
            f"| %{summary['subject_pair_overlap_rate'] * 100:.2f} "
            f"| {summary['weighted_mean_subject_jaccard']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Adil ortak-alt-küme karşılaştırması",
            "",
            (
                "H16 ile KMeans, H16'nın clusterladığı aynı "
                "makaleler üzerinde ayrıca karşılaştırılmıştır."
            ),
            "",
            (
                f"- H16 subject saflığı farkı: "
                f"{subject_purity_difference * 100:+.2f} yüzde puan"
            ),
            (
                f"- H16 ortak-subjectli çift farkı: "
                f"{overlap_difference * 100:+.2f} yüzde puan"
            ),
            "",
            "Pozitif fark H16 lehine, negatif fark KMeans lehinedir.",
            "",
            "## H16 noise özeti",
            "",
            (
                f"- Noise makale: "
                f"{noise_summary['noise_article_count']}"
            ),
            (
                f"- Subject bulunan noise: "
                f"{noise_summary['subject_labeled_noise_count']} "
                f"(%{noise_summary['subject_labeled_noise_rate'] * 100:.2f})"
            ),
            "",
            "### Noise içinde en sık görülen subjectler",
            "",
            "| Sıra | Subject | Makale | Etiketli noise içindeki oran |",
            "|---:|---|---:|---:|",
        ]
    )

    for row in noise_rows[:15]:
        lines.append(
            f"| {row['rank']} "
            f"| {row['subject']} "
            f"| {row['article_count']} "
            f"| %{row['share_of_labeled_noise'] * 100:.2f} |"
        )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as markdown_file:
        markdown_file.write(
            "\n".join(lines)
        )

    return output_path


# =========================================================
# 6. TERMİNAL ÇIKTISI
# =========================================================


def print_results(
    summaries: List[Dict[str, Any]],
    noise_summary: Dict[str, Any],
) -> None:
    """Ana karşılaştırmayı terminalde gösterir."""

    print("\n" + "=" * 125)
    print("KMEANS VE H16 SUBJECT KALİTESİ")
    print("=" * 125)

    header = (
        f"{'Değerlendirme':31}"
        f"{'Cluster':>9}"
        f"{'Atanan':>9}"
        f"{'Kapsam':>10}"
        f"{'Subj. saflık':>15}"
        f"{'Root saflık':>14}"
        f"{'Ortak subj.':>14}"
        f"{'Jaccard':>11}"
    )

    print("\n" + header)
    print("-" * len(header))

    for summary in summaries:
        print(
            f"{summary['evaluation_name']:31}"
            f"{summary['cluster_count']:>9}"
            f"{summary['assigned_article_count']:>9}"
            f"{summary['assigned_coverage'] * 100:>9.1f}%"
            f"{summary['weighted_subject_purity'] * 100:>14.2f}%"
            f"{summary['weighted_root_purity'] * 100:>13.2f}%"
            f"{summary['subject_pair_overlap_rate'] * 100:>13.2f}%"
            f"{summary['weighted_mean_subject_jaccard']:>11.4f}"
        )

    print("\nH16 noise:")
    print(
        f"- Toplam: "
        f"{noise_summary['noise_article_count']}"
    )
    print(
        f"- Subject bulunan: "
        f"{noise_summary['subject_labeled_noise_count']} "
        f"(%{noise_summary['subject_labeled_noise_rate'] * 100:.2f})"
    )


# =========================================================
# 7. MAIN
# =========================================================


def main() -> None:
    print("=" * 80)
    print("DAY 20 — KMEANS VE H16 SUBJECT KALİTESİ")
    print("=" * 80)

    articles = load_articles()

    kmeans_rows = (
        load_kmeans_assignments()
    )

    hdbscan_rows = (
        load_h16_assignments()
    )

    validate_alignment(
        articles=articles,
        kmeans_rows=kmeans_rows,
        hdbscan_rows=hdbscan_rows,
    )

    (
        article_subject_sets,
        article_root_sets,
        subject_display_names,
        root_display_names,
    ) = build_subject_information(
        articles
    )

    kmeans_labels = np.array(
        [
            row["label"]
            for row in kmeans_rows
        ],
        dtype=np.int32,
    )

    hdbscan_labels = np.array(
        [
            row["label"]
            for row in hdbscan_rows
        ],
        dtype=np.int32,
    )

    all_articles_mask = np.ones(
        len(articles),
        dtype=bool,
    )

    h16_clustered_mask = (
        hdbscan_labels >= 0
    )

    summaries: List[
        Dict[str, Any]
    ] = []

    all_detail_rows: List[
        Dict[str, Any]
    ] = []

    (
        kmeans_all_summary,
        kmeans_all_details,
    ) = evaluate_partition(
        evaluation_name=(
            "KMeans_all_articles"
        ),
        labels=kmeans_labels,
        active_mask=all_articles_mask,
        article_subject_sets=(
            article_subject_sets
        ),
        article_root_sets=(
            article_root_sets
        ),
        subject_display_names=(
            subject_display_names
        ),
        root_display_names=(
            root_display_names
        ),
    )

    summaries.append(
        kmeans_all_summary
    )

    all_detail_rows.extend(
        kmeans_all_details
    )

    (
        h16_summary,
        h16_details,
    ) = evaluate_partition(
        evaluation_name=(
            "H16_clustered_articles"
        ),
        labels=hdbscan_labels,
        active_mask=h16_clustered_mask,
        article_subject_sets=(
            article_subject_sets
        ),
        article_root_sets=(
            article_root_sets
        ),
        subject_display_names=(
            subject_display_names
        ),
        root_display_names=(
            root_display_names
        ),
    )

    summaries.append(
        h16_summary
    )

    all_detail_rows.extend(
        h16_details
    )

    (
        kmeans_common_summary,
        kmeans_common_details,
    ) = evaluate_partition(
        evaluation_name=(
            "KMeans_same_H16_subset"
        ),
        labels=kmeans_labels,
        active_mask=h16_clustered_mask,
        article_subject_sets=(
            article_subject_sets
        ),
        article_root_sets=(
            article_root_sets
        ),
        subject_display_names=(
            subject_display_names
        ),
        root_display_names=(
            root_display_names
        ),
    )

    summaries.append(
        kmeans_common_summary
    )

    all_detail_rows.extend(
        kmeans_common_details
    )

    (
        noise_summary,
        noise_distribution_rows,
    ) = analyze_noise_subjects(
        hdbscan_labels=hdbscan_labels,
        article_subject_sets=(
            article_subject_sets
        ),
        article_root_sets=(
            article_root_sets
        ),
        subject_display_names=(
            subject_display_names
        ),
        root_display_names=(
            root_display_names
        ),
    )

    print_results(
        summaries=summaries,
        noise_summary=noise_summary,
    )

    summary_path = save_summary_csv(
        summaries
    )

    detail_path = save_detail_csv(
        all_detail_rows
    )

    noise_path = save_noise_csv(
        noise_distribution_rows
    )

    chart_path = create_quality_chart(
        summaries
    )

    report_path = save_markdown_report(
        summaries=summaries,
        noise_summary=noise_summary,
        noise_rows=(
            noise_distribution_rows
        ),
    )

    print("\n" + "=" * 80)
    print("DAY 20 TAMAMLANDI")
    print("=" * 80)

    print(
        f"\nÖzet CSV:\n"
        f"{summary_path}"
    )

    print(
        f"\nCluster detay CSV:\n"
        f"{detail_path}"
    )

    print(
        f"\nNoise subject dağılımı:\n"
        f"{noise_path}"
    )

    print(
        f"\nKarşılaştırma grafiği:\n"
        f"{chart_path}"
    )

    print(
        f"\nOkunabilir Markdown raporu:\n"
        f"{report_path}"
    )


if __name__ == "__main__":
    main()