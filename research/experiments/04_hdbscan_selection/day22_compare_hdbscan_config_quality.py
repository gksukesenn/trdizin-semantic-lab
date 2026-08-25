import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ARTICLE_COUNT = 1000

CONFIG_IDS = [
    "H01",
    "H16",
    "H18",
]


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


# =========================================================
# 1. VERİLERİ OKUMA
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

    if len(articles) != ARTICLE_COUNT:
        raise ValueError(
            f"{ARTICLE_COUNT} makale bekleniyordu, "
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
            required_columns
            - available_columns
        )

        if missing_columns:
            raise ValueError(
                "KMeans dosyasında eksik sütunlar var: "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            rows.append(
                {
                    "row_index": int(
                        row["row_index"]
                    ),
                    "article_id": row[
                        "article_id"
                    ],
                    "label": int(
                        row["cluster_id"]
                    ),
                }
            )

    rows.sort(
        key=lambda row: row["row_index"]
    )

    if len(rows) != ARTICLE_COUNT:
        raise ValueError(
            f"{ARTICLE_COUNT} KMeans ataması bekleniyordu, "
            f"bulunan: {len(rows)}"
        )

    return rows


def load_hdbscan_assignments() -> List[Dict[str, Any]]:
    """
    Day 17 dosyasından H01, H16 ve H18
    atamalarını birlikte okur.
    """

    input_path = (
        get_project_root()
        / "research" / "outputs"
        / "day17_hdbscan_all_assignments.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"HDBSCAN atama dosyası bulunamadı:\n{input_path}"
        )

    required_columns = {
        "row_index",
        "article_id",
    }

    for config_id in CONFIG_IDS:
        required_columns.add(
            f"{config_id}_label"
        )

        required_columns.add(
            f"{config_id}_probability"
        )

    rows: List[Dict[str, Any]] = []

    with input_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        available_columns = set(
            reader.fieldnames or []
        )

        missing_columns = (
            required_columns
            - available_columns
        )

        if missing_columns:
            raise ValueError(
                "HDBSCAN dosyasında eksik sütunlar var: "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            parsed_row: Dict[str, Any] = {
                "row_index": int(
                    row["row_index"]
                ),
                "article_id": row[
                    "article_id"
                ],
            }

            for config_id in CONFIG_IDS:
                parsed_row[
                    f"{config_id}_label"
                ] = int(
                    row[f"{config_id}_label"]
                )

                parsed_row[
                    f"{config_id}_probability"
                ] = float(
                    row[
                        f"{config_id}_probability"
                    ]
                )

            rows.append(parsed_row)

    rows.sort(
        key=lambda row: row["row_index"]
    )

    if len(rows) != ARTICLE_COUNT:
        raise ValueError(
            f"{ARTICLE_COUNT} HDBSCAN ataması bekleniyordu, "
            f"bulunan: {len(rows)}"
        )

    return rows


def validate_alignment(
    articles: List[Dict[str, Any]],
    kmeans_rows: List[Dict[str, Any]],
    hdbscan_rows: List[Dict[str, Any]],
) -> None:
    """Bütün dosyaların aynı makale sırasına sahip olduğunu doğrular."""

    for row_index in range(
        ARTICLE_COUNT
    ):
        article_id = str(
            articles[row_index].get(
                "article_id",
                "",
            )
        )

        if (
            kmeans_rows[row_index][
                "row_index"
            ]
            != row_index
        ):
            raise ValueError(
                f"KMeans row_index uyuşmazlığı: {row_index}"
            )

        if (
            hdbscan_rows[row_index][
                "row_index"
            ]
            != row_index
        ):
            raise ValueError(
                f"HDBSCAN row_index uyuşmazlığı: {row_index}"
            )

        if (
            kmeans_rows[row_index][
                "article_id"
            ]
            != article_id
        ):
            raise ValueError(
                f"KMeans article_id uyuşmazlığı: {row_index}"
            )

        if (
            hdbscan_rows[row_index][
                "article_id"
            ]
            != article_id
        ):
            raise ValueError(
                f"HDBSCAN article_id uyuşmazlığı: {row_index}"
            )


# =========================================================
# 2. SUBJECT METADATA
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

    full_name = subject.get(
        "fullName"
    )

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

    root_id = subject.get(
        "rootId"
    )

    if root_id is not None:
        return f"root-id:{root_id}"

    root_name = subject.get(
        "rootName"
    )

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
    """Subjectin okunabilir adını döndürür."""

    full_name = subject.get(
        "fullName"
    )

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
    """Kök alanın okunabilir adını döndürür."""

    root_name = subject.get(
        "rootName"
    )

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
    """Her makalenin subject ve root kümelerini oluşturur."""

    article_subject_sets: List[
        Set[str]
    ] = []

    article_root_sets: List[
        Set[str]
    ] = []

    subject_display_names: Dict[
        str,
        str
    ] = {}

    root_display_names: Dict[
        str,
        str
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

            subject_key = (
                create_subject_key(
                    subject
                )
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
# 3. KALİTE METRİKLERİ
# =========================================================


def calculate_pair_metrics(
    subject_sets: List[Set[str]],
) -> Dict[str, Any]:
    """
    Aynı cluster içindeki etiketli makale çiftlerinin
    subject benzerliğini hesaplar.
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

            union = (
                first_set
                | second_set
            )

            if not union:
                continue

            intersection = (
                first_set
                & second_set
            )

            pair_count += 1

            jaccard_total += (
                len(intersection)
                / len(union)
            )

            if intersection:
                overlap_count += 1

    return {
        "pair_count": pair_count,
        "jaccard_total": jaccard_total,
        "overlap_count": overlap_count,
        "mean_jaccard": (
            jaccard_total
            / pair_count
            if pair_count
            else None
        ),
        "overlap_rate": (
            overlap_count
            / pair_count
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
    """Bir clustering çözümünün subject kalitesini ölçer."""

    active_indices = np.where(
        active_mask
    )[0]

    cluster_ids = sorted(
        {
            int(labels[index])
            for index in active_indices
            if int(labels[index]) >= 0
        }
    )

    detail_rows: List[
        Dict[str, Any]
    ] = []

    total_labeled_count = 0
    total_root_labeled_count = 0

    dominant_subject_total = 0
    dominant_root_total = 0

    subject_purities: List[
        float
    ] = []

    root_purities: List[
        float
    ] = []

    total_pair_count = 0
    total_jaccard = 0.0
    total_overlap_count = 0

    for cluster_id in cluster_ids:
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
            dominant_subject_count = 0
            subject_purity = None
            dominant_subject_name = ""

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
            dominant_root_count = 0
            root_purity = None
            dominant_root_name = ""

        pair_metrics = (
            calculate_pair_metrics(
                [
                    article_subject_sets[
                        index
                    ]
                    for index
                    in subject_labeled_indices
                ]
            )
        )

        total_labeled_count += len(
            subject_labeled_indices
        )

        total_root_labeled_count += len(
            root_labeled_indices
        )

        dominant_subject_total += (
            dominant_subject_count
        )

        dominant_root_total += (
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
            pair_metrics[
                "jaccard_total"
            ]
        )

        total_overlap_count += (
            pair_metrics[
                "overlap_count"
            ]
        )

        detail_rows.append(
            {
                "evaluation_name": (
                    evaluation_name
                ),
                "cluster_id": (
                    cluster_id
                ),
                "cluster_size": len(
                    cluster_indices
                ),
                "subject_labeled_count": len(
                    subject_labeled_indices
                ),
                "dominant_subject": (
                    dominant_subject_name
                ),
                "subject_purity": (
                    subject_purity
                ),
                "dominant_root": (
                    dominant_root_name
                ),
                "root_purity": (
                    root_purity
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

    assigned_count = int(
        len(active_indices)
    )

    summary = {
        "evaluation_name": (
            evaluation_name
        ),
        "cluster_count": len(
            cluster_ids
        ),
        "assigned_article_count": (
            assigned_count
        ),
        "assigned_coverage": (
            assigned_count
            / ARTICLE_COUNT
        ),
        "noise_count": (
            ARTICLE_COUNT
            - assigned_count
        ),
        "subject_labeled_article_count": (
            total_labeled_count
        ),
        "subject_labeled_rate": (
            total_labeled_count
            / assigned_count
            if assigned_count
            else 0.0
        ),
        "weighted_subject_purity": (
            dominant_subject_total
            / total_labeled_count
            if total_labeled_count
            else 0.0
        ),
        "macro_subject_purity": (
            float(
                np.mean(
                    subject_purities
                )
            )
            if subject_purities
            else 0.0
        ),
        "weighted_root_purity": (
            dominant_root_total
            / total_root_labeled_count
            if total_root_labeled_count
            else 0.0
        ),
        "macro_root_purity": (
            float(
                np.mean(
                    root_purities
                )
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
# 4. KARŞILAŞTIRMA TABLOLARI
# =========================================================


def build_decision_rows(
    summary_by_name: Dict[
        str,
        Dict[str, Any],
    ],
) -> List[Dict[str, Any]]:
    """Her HDBSCAN ayarının KMeans'e karşı farkını hesaplar."""

    decision_rows: List[
        Dict[str, Any]
    ] = []

    for config_id in CONFIG_IDS:
        own_summary = summary_by_name[
            f"{config_id}_own_subset"
        ]

        kmeans_summary = summary_by_name[
            f"KMeans_same_{config_id}_subset"
        ]

        common_summary = summary_by_name[
            f"{config_id}_common_subset"
        ]

        decision_rows.append(
            {
                "config_id": config_id,
                "cluster_count": own_summary[
                    "cluster_count"
                ],
                "assigned_article_count": own_summary[
                    "assigned_article_count"
                ],
                "coverage": own_summary[
                    "assigned_coverage"
                ],
                "noise_count": own_summary[
                    "noise_count"
                ],
                "own_subject_purity": own_summary[
                    "weighted_subject_purity"
                ],
                "kmeans_same_subset_subject_purity": (
                    kmeans_summary[
                        "weighted_subject_purity"
                    ]
                ),
                "subject_purity_gain_vs_kmeans": (
                    own_summary[
                        "weighted_subject_purity"
                    ]
                    - kmeans_summary[
                        "weighted_subject_purity"
                    ]
                ),
                "own_overlap_rate": own_summary[
                    "subject_pair_overlap_rate"
                ],
                "kmeans_same_subset_overlap_rate": (
                    kmeans_summary[
                        "subject_pair_overlap_rate"
                    ]
                ),
                "overlap_gain_vs_kmeans": (
                    own_summary[
                        "subject_pair_overlap_rate"
                    ]
                    - kmeans_summary[
                        "subject_pair_overlap_rate"
                    ]
                ),
                "own_jaccard": own_summary[
                    "weighted_mean_subject_jaccard"
                ],
                "kmeans_same_subset_jaccard": (
                    kmeans_summary[
                        "weighted_mean_subject_jaccard"
                    ]
                ),
                "jaccard_gain_vs_kmeans": (
                    own_summary[
                        "weighted_mean_subject_jaccard"
                    ]
                    - kmeans_summary[
                        "weighted_mean_subject_jaccard"
                    ]
                ),
                "common_subset_article_count": (
                    common_summary[
                        "assigned_article_count"
                    ]
                ),
                "common_subset_subject_purity": (
                    common_summary[
                        "weighted_subject_purity"
                    ]
                ),
                "common_subset_overlap_rate": (
                    common_summary[
                        "subject_pair_overlap_rate"
                    ]
                ),
                "common_subset_jaccard": (
                    common_summary[
                        "weighted_mean_subject_jaccard"
                    ]
                ),
            }
        )

    return decision_rows


def print_overall_table(
    summaries: List[Dict[str, Any]],
) -> None:
    """Kapsama ve kalite sonuçlarını terminalde gösterir."""

    wanted_names = [
        "KMeans_all_articles",
        "H01_own_subset",
        "H16_own_subset",
        "H18_own_subset",
    ]

    summary_by_name = {
        summary["evaluation_name"]:
        summary
        for summary in summaries
    }

    print("\n" + "=" * 120)
    print("GENEL KAPSAMA VE KALİTE")
    print("=" * 120)

    header = (
        f"{'Yöntem':25}"
        f"{'Cluster':>9}"
        f"{'Atanan':>9}"
        f"{'Kapsam':>10}"
        f"{'Subj. saflık':>15}"
        f"{'Ortak subj.':>14}"
        f"{'Jaccard':>11}"
    )

    print("\n" + header)
    print("-" * len(header))

    for name in wanted_names:
        summary = summary_by_name[
            name
        ]

        print(
            f"{name:25}"
            f"{summary['cluster_count']:>9}"
            f"{summary['assigned_article_count']:>9}"
            f"{summary['assigned_coverage'] * 100:>9.1f}%"
            f"{summary['weighted_subject_purity'] * 100:>14.2f}%"
            f"{summary['subject_pair_overlap_rate'] * 100:>13.2f}%"
            f"{summary['weighted_mean_subject_jaccard']:>11.4f}"
        )


def print_paired_table(
    decision_rows: List[Dict[str, Any]],
) -> None:
    """Her HDBSCAN ayarını aynı makalelerdeki KMeans ile karşılaştırır."""

    print("\n" + "=" * 125)
    print("AYNI MAKALELERDE HDBSCAN — KMEANS FARKI")
    print("=" * 125)

    header = (
        f"{'ID':5}"
        f"{'Kapsam':>10}"
        f"{'HDB saflık':>13}"
        f"{'KM saflık':>12}"
        f"{'Saflık farkı':>14}"
        f"{'HDB ortak':>12}"
        f"{'KM ortak':>11}"
        f"{'Ortak farkı':>13}"
        f"{'Jaccard farkı':>15}"
    )

    print("\n" + header)
    print("-" * len(header))

    for row in decision_rows:
        print(
            f"{row['config_id']:5}"
            f"{row['coverage'] * 100:>9.1f}%"
            f"{row['own_subject_purity'] * 100:>12.2f}%"
            f"{row['kmeans_same_subset_subject_purity'] * 100:>11.2f}%"
            f"{row['subject_purity_gain_vs_kmeans'] * 100:>+13.2f}"
            f"{row['own_overlap_rate'] * 100:>11.2f}%"
            f"{row['kmeans_same_subset_overlap_rate'] * 100:>10.2f}%"
            f"{row['overlap_gain_vs_kmeans'] * 100:>+12.2f}"
            f"{row['jaccard_gain_vs_kmeans']:>+15.4f}"
        )


def print_common_subset_table(
    decision_rows: List[Dict[str, Any]],
) -> None:
    """
    H01, H16 ve H18'in üçü tarafından da clusterlanan
    aynı makalelerdeki kaliteyi gösterir.
    """

    print("\n" + "=" * 105)
    print("ÜÇ HDBSCAN AYARININ ORTAK MAKALE ALT KÜMESİ")
    print("=" * 105)

    common_count = decision_rows[
        0
    ]["common_subset_article_count"]

    print(
        f"\nÜç ayarın da clusterladığı makale: "
        f"{common_count}"
    )

    header = (
        f"{'ID':5}"
        f"{'Subject saflığı':>19}"
        f"{'Ortak subject':>18}"
        f"{'Jaccard':>13}"
    )

    print("\n" + header)
    print("-" * len(header))

    for row in decision_rows:
        print(
            f"{row['config_id']:5}"
            f"{row['common_subset_subject_purity'] * 100:>18.2f}%"
            f"{row['common_subset_overlap_rate'] * 100:>17.2f}%"
            f"{row['common_subset_jaccard']:>13.4f}"
        )


# =========================================================
# 5. DOSYALAR VE GÖRSEL
# =========================================================


def save_summary_csv(
    summaries: List[Dict[str, Any]],
) -> Path:
    """Bütün değerlendirmeleri CSV olarak kaydeder."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day22_hdbscan_quality_all_evaluations.csv"
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
    """Cluster bazlı değerlendirmeleri kaydeder."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day22_hdbscan_quality_cluster_details.csv"
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


def save_decision_csv(
    decision_rows: List[Dict[str, Any]],
) -> Path:
    """Ana HDBSCAN seçimi için özet tabloyu kaydeder."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day22_hdbscan_config_decision.csv"
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                decision_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            decision_rows
        )

    return output_path


def create_coverage_quality_chart(
    summaries: List[Dict[str, Any]],
) -> Path:
    """Kapsama ve subject saflığı ilişkisini gösterir."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day22_hdbscan_coverage_vs_quality.png"
    )

    wanted_names = [
        "KMeans_all_articles",
        "H01_own_subset",
        "H16_own_subset",
        "H18_own_subset",
    ]

    summary_by_name = {
        summary["evaluation_name"]:
        summary
        for summary in summaries
    }

    plt.figure(
        figsize=(11, 7)
    )

    for name in wanted_names:
        summary = summary_by_name[
            name
        ]

        x_value = (
            summary[
                "assigned_coverage"
            ]
            * 100
        )

        y_value = (
            summary[
                "weighted_subject_purity"
            ]
            * 100
        )

        plt.scatter(
            x_value,
            y_value,
            s=100,
        )

        plt.annotate(
            name,
            (
                x_value,
                y_value,
            ),
            xytext=(6, 6),
            textcoords="offset points",
        )

    plt.title(
        "HDBSCAN Ayarları: Kapsama ve Subject Saflığı"
    )

    plt.xlabel(
        "Clusterlanan makale oranı (%)"
    )

    plt.ylabel(
        "Ağırlıklı subject saflığı (%)"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=170,
    )

    plt.close()

    return output_path


def save_markdown_report(
    summaries: List[Dict[str, Any]],
    decision_rows: List[Dict[str, Any]],
) -> Path:
    """Sonuçları okunabilir Markdown raporuna kaydeder."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day22_hdbscan_config_comparison_report.md"
    )

    summary_by_name = {
        summary["evaluation_name"]:
        summary
        for summary in summaries
    }

    lines: List[str] = [
        "# HDBSCAN Ana Ayar Karşılaştırması",
        "",
        (
            "Subjectler embedding, UMAP veya clustering "
            "girdisi olarak kullanılmamıştır."
        ),
        "",
        (
            "Subject metadata yalnızca clustering sonuçlarını "
            "sonradan değerlendirmek için kullanılmıştır."
        ),
        "",
        "## Genel kapsama ve kalite",
        "",
        (
            "| Yöntem | Cluster | Atanan | Kapsam | "
            "Subject saflığı | Ortak subjectli çift | Jaccard |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for name in [
        "KMeans_all_articles",
        "H01_own_subset",
        "H16_own_subset",
        "H18_own_subset",
    ]:
        summary = summary_by_name[
            name
        ]

        lines.append(
            f"| {name} "
            f"| {summary['cluster_count']} "
            f"| {summary['assigned_article_count']} "
            f"| %{summary['assigned_coverage'] * 100:.2f} "
            f"| %{summary['weighted_subject_purity'] * 100:.2f} "
            f"| %{summary['subject_pair_overlap_rate'] * 100:.2f} "
            f"| {summary['weighted_mean_subject_jaccard']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Aynı makalelerde KMeans farkı",
            "",
            (
                "| Ayar | Kapsam | Subject saflığı farkı | "
                "Ortak subject farkı | Jaccard farkı |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )

    for row in decision_rows:
        lines.append(
            f"| {row['config_id']} "
            f"| %{row['coverage'] * 100:.2f} "
            f"| {row['subject_purity_gain_vs_kmeans'] * 100:+.2f} puan "
            f"| {row['overlap_gain_vs_kmeans'] * 100:+.2f} puan "
            f"| {row['jaccard_gain_vs_kmeans']:+.4f} |"
        )

    common_count = decision_rows[
        0
    ]["common_subset_article_count"]

    lines.extend(
        [
            "",
            "## Üç ayarın ortak makale alt kümesi",
            "",
            (
                f"Üç HDBSCAN ayarının da clusterladığı "
                f"makale sayısı: {common_count}"
            ),
            "",
            "| Ayar | Subject saflığı | Ortak subject | Jaccard |",
            "|---|---:|---:|---:|",
        ]
    )

    for row in decision_rows:
        lines.append(
            f"| {row['config_id']} "
            f"| %{row['common_subset_subject_purity'] * 100:.2f} "
            f"| %{row['common_subset_overlap_rate'] * 100:.2f} "
            f"| {row['common_subset_jaccard']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Karar ölçütü",
            "",
            (
                "Ana HDBSCAN ayarı yalnızca en yüksek saflığa "
                "göre seçilmeyecektir."
            ),
            "",
            (
                "Kapsama, aynı-alt-küme kalite farkı, common-subset "
                "kalitesi ve noise oranı birlikte değerlendirilecektir."
            ),
        ]
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
# 6. MAIN
# =========================================================


def main() -> None:
    print("=" * 85)
    print("DAY 22 — HDBSCAN ANA AYAR KARŞILAŞTIRMASI")
    print("=" * 85)

    articles = load_articles()

    kmeans_rows = (
        load_kmeans_assignments()
    )

    hdbscan_rows = (
        load_hdbscan_assignments()
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

    hdbscan_labels: Dict[
        str,
        np.ndarray,
    ] = {}

    for config_id in CONFIG_IDS:
        hdbscan_labels[
            config_id
        ] = np.array(
            [
                row[
                    f"{config_id}_label"
                ]
                for row in hdbscan_rows
            ],
            dtype=np.int32,
        )

    all_articles_mask = np.ones(
        ARTICLE_COUNT,
        dtype=bool,
    )

    common_hdbscan_mask = np.ones(
        ARTICLE_COUNT,
        dtype=bool,
    )

    for config_id in CONFIG_IDS:
        common_hdbscan_mask = (
            common_hdbscan_mask
            & (
                hdbscan_labels[
                    config_id
                ]
                >= 0
            )
        )

    summaries: List[
        Dict[str, Any]
    ] = []

    detail_rows: List[
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

    detail_rows.extend(
        kmeans_all_details
    )

    for config_id in CONFIG_IDS:
        config_labels = (
            hdbscan_labels[
                config_id
            ]
        )

        own_mask = (
            config_labels >= 0
        )

        (
            hdbscan_summary,
            hdbscan_details,
        ) = evaluate_partition(
            evaluation_name=(
                f"{config_id}_own_subset"
            ),
            labels=config_labels,
            active_mask=own_mask,
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
            hdbscan_summary
        )

        detail_rows.extend(
            hdbscan_details
        )

        (
            kmeans_same_summary,
            kmeans_same_details,
        ) = evaluate_partition(
            evaluation_name=(
                f"KMeans_same_{config_id}_subset"
            ),
            labels=kmeans_labels,
            active_mask=own_mask,
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
            kmeans_same_summary
        )

        detail_rows.extend(
            kmeans_same_details
        )

        (
            common_summary,
            common_details,
        ) = evaluate_partition(
            evaluation_name=(
                f"{config_id}_common_subset"
            ),
            labels=config_labels,
            active_mask=common_hdbscan_mask,
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
            common_summary
        )

        detail_rows.extend(
            common_details
        )

    (
        kmeans_common_summary,
        kmeans_common_details,
    ) = evaluate_partition(
        evaluation_name=(
            "KMeans_common_HDBSCAN_subset"
        ),
        labels=kmeans_labels,
        active_mask=common_hdbscan_mask,
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

    detail_rows.extend(
        kmeans_common_details
    )

    summary_by_name = {
        summary["evaluation_name"]:
        summary
        for summary in summaries
    }

    decision_rows = (
        build_decision_rows(
            summary_by_name
        )
    )

    print_overall_table(
        summaries
    )

    print_paired_table(
        decision_rows
    )

    print_common_subset_table(
        decision_rows
    )

    summary_path = save_summary_csv(
        summaries
    )

    detail_path = save_detail_csv(
        detail_rows
    )

    decision_path = save_decision_csv(
        decision_rows
    )

    chart_path = (
        create_coverage_quality_chart(
            summaries
        )
    )

    report_path = (
        save_markdown_report(
            summaries=summaries,
            decision_rows=(
                decision_rows
            ),
        )
    )

    print("\n" + "=" * 85)
    print("DAY 22 TAMAMLANDI")
    print("=" * 85)

    print(
        f"\nBütün değerlendirmeler:\n"
        f"{summary_path}"
    )

    print(
        f"\nCluster detayları:\n"
        f"{detail_path}"
    )

    print(
        f"\nAna karar tablosu:\n"
        f"{decision_path}"
    )

    print(
        f"\nKapsama-kalite görseli:\n"
        f"{chart_path}"
    )

    print(
        f"\nOkunabilir rapor:\n"
        f"{report_path}"
    )


if __name__ == "__main__":
    main()