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
HDBSCAN_CONFIG_ID = "H01"

CORE_K_VALUES = [
    5,
    10,
]

METHOD_DISPLAY_NAMES = {
    "medoid": "Tek medoid",
    "centroid": "Cluster centroidi",
    "top5_core_mean": "Top-5 çekirdek ortalaması",
    "top10_core_mean": "Top-10 çekirdek ortalaması",
}


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


def get_output_directory() -> Path:
    """Çıktı klasörünü hazırlar."""

    output_directory = (
        get_project_root()
        / "research" / "outputs"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory


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
                article = json.loads(
                    cleaned_line
                )
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL satırı okunamadı: "
                    f"{line_number}"
                ) from error

            if isinstance(article, dict):
                articles.append(article)

    if len(articles) != ARTICLE_COUNT:
        raise ValueError(
            f"{ARTICLE_COUNT} makale bekleniyordu, "
            f"bulunan: {len(articles)}"
        )

    return articles


def normalize_embedding_rows(
    embeddings: np.ndarray,
) -> np.ndarray:
    """Vektörleri birim uzunluğa getirir."""

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


def load_embeddings() -> np.ndarray:
    """TR-MTEB embedding matrisini yükler."""

    input_path = (
        get_project_root()
        / "research" / "outputs"
        / "day13_embeddings"
        / "tr_mteb.npy"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Embedding dosyası bulunamadı:\n"
            f"{input_path}"
        )

    embeddings = np.load(
        input_path
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Embedding matrisi iki boyutlu değil: "
            f"{embeddings.shape}"
        )

    if embeddings.shape[0] != ARTICLE_COUNT:
        raise ValueError(
            f"{ARTICLE_COUNT} embedding bekleniyordu, "
            f"bulunan: {embeddings.shape[0]}"
        )

    if not np.isfinite(
        embeddings
    ).all():
        raise ValueError(
            "Embedding matrisinde NaN veya "
            "sonsuz değer var."
        )

    return normalize_embedding_rows(
        embeddings
    )


def load_h01_assignments() -> List[Dict[str, Any]]:
    """Day 17 dosyasından H01 atamalarını okur."""

    input_path = (
        get_project_root()
        / "research" / "outputs"
        / "day17_hdbscan_all_assignments.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"HDBSCAN atama dosyası bulunamadı:\n"
            f"{input_path}"
        )

    label_column = (
        f"{HDBSCAN_CONFIG_ID}_label"
    )

    probability_column = (
        f"{HDBSCAN_CONFIG_ID}_probability"
    )

    outlier_column = (
        f"{HDBSCAN_CONFIG_ID}_outlier_score"
    )

    required_columns = {
        "row_index",
        "article_id",
        label_column,
        probability_column,
        outlier_column,
    }

    rows: List[Dict[str, Any]] = []

    with input_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file
        )

        available_columns = set(
            reader.fieldnames or []
        )

        missing_columns = (
            required_columns
            - available_columns
        )

        if missing_columns:
            raise ValueError(
                "HDBSCAN dosyasında eksik "
                "sütunlar var: "
                + ", ".join(
                    sorted(missing_columns)
                )
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
                        row[label_column]
                    ),
                    "probability": float(
                        row[
                            probability_column
                        ]
                    ),
                    "outlier_score": float(
                        row[outlier_column]
                    ),
                }
            )

    rows.sort(
        key=lambda row: row["row_index"]
    )

    if len(rows) != ARTICLE_COUNT:
        raise ValueError(
            f"{ARTICLE_COUNT} HDBSCAN ataması "
            f"bekleniyordu, bulunan: {len(rows)}"
        )

    return rows


def validate_alignment(
    articles: List[Dict[str, Any]],
    assignments: List[Dict[str, Any]],
) -> None:
    """Makale ve atama satırlarının eşleştiğini doğrular."""

    for row_index in range(
        ARTICLE_COUNT
    ):
        article_id = str(
            articles[row_index].get(
                "article_id",
                "",
            )
        )

        assignment = assignments[
            row_index
        ]

        if (
            assignment["row_index"]
            != row_index
        ):
            raise ValueError(
                f"row_index uyuşmazlığı: "
                f"{row_index}"
            )

        if (
            assignment["article_id"]
            != article_id
        ):
            raise ValueError(
                f"article_id uyuşmazlığı: "
                f"{row_index}"
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

        if isinstance(
            parsed_value,
            dict,
        ):
            return parsed_value

    return None


def create_subject_key(
    subject: Dict[str, Any],
) -> Optional[str]:
    """Subject için kararlı anahtar üretir."""

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


def build_subject_information(
    articles: List[Dict[str, Any]],
) -> Tuple[
    List[Set[str]],
    Dict[str, str],
]:
    """Makalelerin subject kümelerini hazırlar."""

    article_subject_sets: List[
        Set[str]
    ] = []

    subject_display_names: Dict[
        str,
        str,
    ] = {}

    for article in articles:
        subject_keys: Set[str] = set()

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

            if not subject_key:
                continue

            full_name = subject.get(
                "fullName"
            )

            name = subject.get("name")

            if (
                isinstance(full_name, str)
                and full_name.strip()
            ):
                display_name = (
                    full_name.strip()
                )
            elif (
                isinstance(name, str)
                and name.strip()
            ):
                display_name = (
                    name.strip()
                )
            else:
                display_name = (
                    subject_key
                )

            subject_keys.add(
                subject_key
            )

            subject_display_names.setdefault(
                subject_key,
                display_name,
            )

        article_subject_sets.append(
            subject_keys
        )

    return (
        article_subject_sets,
        subject_display_names,
    )


# =========================================================
# 3. CLUSTER TEMSİLLERİ
# =========================================================


def get_cluster_ids(
    assignments: List[Dict[str, Any]],
) -> List[int]:
    """Noise dışındaki H01 cluster ID'lerini döndürür."""

    return sorted(
        {
            int(row["label"])
            for row in assignments
            if int(row["label"]) >= 0
        }
    )


def find_medoid_index(
    cluster_indices: List[int],
    embeddings: np.ndarray,
) -> int:
    """
    Cluster içindeki ortalama cosine benzerliği
    en yüksek gerçek makaleyi bulur.
    """

    cluster_vectors = embeddings[
        cluster_indices
    ]

    similarity_matrix = (
        cluster_vectors
        @ cluster_vectors.T
    )

    mean_similarities = (
        similarity_matrix.mean(
            axis=1
        )
    )

    best_local_index = int(
        np.argmax(
            mean_similarities
        )
    )

    return int(
        cluster_indices[
            best_local_index
        ]
    )


def calculate_normalized_centroid(
    cluster_indices: List[int],
    embeddings: np.ndarray,
) -> np.ndarray:
    """Clusterın normalize edilmiş ortalama vektörünü hesaplar."""

    centroid = embeddings[
        cluster_indices
    ].mean(
        axis=0
    )

    centroid_norm = float(
        np.linalg.norm(
            centroid
        )
    )

    if centroid_norm == 0:
        return centroid.astype(
            np.float32,
            copy=False,
        )

    return (
        centroid
        / centroid_norm
    ).astype(
        np.float32,
        copy=False,
    )


def select_core_indices(
    cluster_indices: List[int],
    assignments: List[Dict[str, Any]],
    core_size: int,
) -> List[int]:
    """
    HDBSCAN üyelik olasılığı en yüksek makaleleri
    clusterın güçlü çekirdek üyeleri olarak seçer.
    """

    ranked_indices = sorted(
        cluster_indices,
        key=lambda article_index: (
            -assignments[
                article_index
            ]["probability"],
            assignments[
                article_index
            ]["outlier_score"],
            assignments[
                article_index
            ]["article_id"],
        ),
    )

    return ranked_indices[
        :min(
            core_size,
            len(ranked_indices),
        )
    ]


def calculate_core_mean_vector(
    core_indices: List[int],
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Güçlü çekirdek makalelerin ortalama vektörünü döndürür.

    Bu vektör özellikle normalize edilmez.

    Normalize edilmiş bir sorgu ile bu vektörün
    çarpımı, çekirdek üyelerle ortalama cosine
    benzerliğe eşittir.
    """

    return embeddings[
        core_indices
    ].mean(
        axis=0
    ).astype(
        np.float32,
        copy=False,
    )


def find_dominant_subject(
    cluster_indices: List[int],
    article_subject_sets: List[Set[str]],
    subject_display_names: Dict[str, str],
) -> Tuple[
    str,
    str,
    int,
    int,
    float,
]:
    """Clusterın baskın subjectini hesaplar."""

    subject_counter = Counter()

    labeled_article_count = 0

    for article_index in cluster_indices:
        subject_set = (
            article_subject_sets[
                article_index
            ]
        )

        if not subject_set:
            continue

        labeled_article_count += 1

        for subject_key in subject_set:
            subject_counter[
                subject_key
            ] += 1

    if not subject_counter:
        return (
            "",
            "",
            0,
            0,
            0.0,
        )

    (
        dominant_subject_key,
        dominant_subject_count,
    ) = subject_counter.most_common(
        1
    )[0]

    dominant_subject_name = (
        subject_display_names.get(
            dominant_subject_key,
            dominant_subject_key,
        )
    )

    subject_purity = (
        dominant_subject_count
        / labeled_article_count
        if labeled_article_count
        else 0.0
    )

    return (
        dominant_subject_key,
        dominant_subject_name,
        int(dominant_subject_count),
        labeled_article_count,
        float(subject_purity),
    )


def build_cluster_representations(
    articles: List[Dict[str, Any]],
    embeddings: np.ndarray,
    assignments: List[Dict[str, Any]],
    article_subject_sets: List[Set[str]],
    subject_display_names: Dict[str, str],
) -> Tuple[
    List[int],
    Dict[int, Dict[str, Any]],
    Dict[str, np.ndarray],
]:
    """Dört farklı cluster temsil yöntemini oluşturur."""

    cluster_ids = get_cluster_ids(
        assignments
    )

    cluster_info: Dict[
        int,
        Dict[str, Any],
    ] = {}

    medoid_vectors: List[
        np.ndarray
    ] = []

    centroid_vectors: List[
        np.ndarray
    ] = []

    core_vectors_by_size: Dict[
        int,
        List[np.ndarray],
    ] = {
        core_size: []
        for core_size in CORE_K_VALUES
    }

    for cluster_id in cluster_ids:
        cluster_indices = [
            article_index
            for article_index, assignment
            in enumerate(assignments)
            if assignment["label"]
            == cluster_id
        ]

        medoid_index = (
            find_medoid_index(
                cluster_indices=(
                    cluster_indices
                ),
                embeddings=embeddings,
            )
        )

        centroid_vector = (
            calculate_normalized_centroid(
                cluster_indices=(
                    cluster_indices
                ),
                embeddings=embeddings,
            )
        )

        (
            dominant_subject_key,
            dominant_subject_name,
            dominant_subject_count,
            labeled_article_count,
            subject_purity,
        ) = find_dominant_subject(
            cluster_indices=(
                cluster_indices
            ),
            article_subject_sets=(
                article_subject_sets
            ),
            subject_display_names=(
                subject_display_names
            ),
        )

        medoid_vectors.append(
            embeddings[
                medoid_index
            ]
        )

        centroid_vectors.append(
            centroid_vector
        )

        core_indices_by_size: Dict[
            int,
            List[int],
        ] = {}

        for core_size in CORE_K_VALUES:
            core_indices = (
                select_core_indices(
                    cluster_indices=(
                        cluster_indices
                    ),
                    assignments=(
                        assignments
                    ),
                    core_size=core_size,
                )
            )

            core_indices_by_size[
                core_size
            ] = core_indices

            core_vectors_by_size[
                core_size
            ].append(
                calculate_core_mean_vector(
                    core_indices=(
                        core_indices
                    ),
                    embeddings=embeddings,
                )
            )

        cluster_info[
            cluster_id
        ] = {
            "cluster_id": cluster_id,
            "cluster_size": len(
                cluster_indices
            ),
            "cluster_indices": (
                cluster_indices
            ),
            "dominant_subject_key": (
                dominant_subject_key
            ),
            "dominant_subject_name": (
                dominant_subject_name
                or f"H01 Cluster {cluster_id}"
            ),
            "dominant_subject_count": (
                dominant_subject_count
            ),
            "subject_labeled_count": (
                labeled_article_count
            ),
            "subject_purity": (
                subject_purity
            ),
            "medoid_index": (
                medoid_index
            ),
            "medoid_article_id": articles[
                medoid_index
            ].get(
                "article_id",
                "",
            ),
            "medoid_title": articles[
                medoid_index
            ].get(
                "title_tr",
                "",
            ),
            "core_indices_by_size": (
                core_indices_by_size
            ),
        }

    representations = {
        "medoid": np.vstack(
            medoid_vectors
        ).astype(
            np.float32,
            copy=False,
        ),
        "centroid": np.vstack(
            centroid_vectors
        ).astype(
            np.float32,
            copy=False,
        ),
    }

    for core_size in CORE_K_VALUES:
        representations[
            f"top{core_size}_core_mean"
        ] = np.vstack(
            core_vectors_by_size[
                core_size
            ]
        ).astype(
            np.float32,
            copy=False,
        )

    return (
        cluster_ids,
        cluster_info,
        representations,
    )


# =========================================================
# 4. YÖNTEMLERİ DEĞERLENDİRME
# =========================================================


def rank_cluster_positions(
    score_row: np.ndarray,
) -> Tuple[int, int]:
    """En yüksek ve ikinci en yüksek cluster konumunu bulur."""

    ranked_positions = np.argsort(
        score_row
    )[::-1]

    return (
        int(ranked_positions[0]),
        int(ranked_positions[1]),
    )


def evaluate_assignment_method(
    method_name: str,
    representation_matrix: np.ndarray,
    cluster_ids: List[int],
    cluster_info: Dict[int, Dict[str, Any]],
    articles: List[Dict[str, Any]],
    embeddings: np.ndarray,
    assignments: List[Dict[str, Any]],
    article_subject_sets: List[Set[str]],
    subject_display_names: Dict[str, str],
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    """
    Bir temsil yöntemini noise makaleler ve
    doğrudan H01 makaleleri üzerinde değerlendirir.
    """

    score_matrix = (
        embeddings
        @ representation_matrix.T
    )

    noise_indices = [
        article_index
        for article_index, assignment
        in enumerate(assignments)
        if assignment["label"] == -1
    ]

    clustered_indices = [
        article_index
        for article_index, assignment
        in enumerate(assignments)
        if assignment["label"] >= 0
    ]

    # H01 clusterlarının temsillerle yeniden
    # bulunabilme oranı.
    recovery_match_count = 0

    for article_index in (
        clustered_indices
    ):
        predicted_position = int(
            np.argmax(
                score_matrix[
                    article_index
                ]
            )
        )

        predicted_cluster_id = (
            cluster_ids[
                predicted_position
            ]
        )

        if (
            predicted_cluster_id
            == assignments[
                article_index
            ]["label"]
        ):
            recovery_match_count += 1

    cluster_recovery_rate = (
        recovery_match_count
        / len(clustered_indices)
        if clustered_indices
        else 0.0
    )

    detail_rows: List[
        Dict[str, Any]
    ] = []

    labeled_noise_count = 0
    top1_match_count = 0
    top2_match_count = 0

    margins: List[float] = []

    for article_index in noise_indices:
        score_row = score_matrix[
            article_index
        ]

        (
            primary_position,
            secondary_position,
        ) = rank_cluster_positions(
            score_row
        )

        primary_cluster_id = (
            cluster_ids[
                primary_position
            ]
        )

        secondary_cluster_id = (
            cluster_ids[
                secondary_position
            ]
        )

        primary_score = float(
            score_row[
                primary_position
            ]
        )

        secondary_score = float(
            score_row[
                secondary_position
            ]
        )

        score_margin = (
            primary_score
            - secondary_score
        )

        margins.append(
            score_margin
        )

        primary_subject_key = (
            cluster_info[
                primary_cluster_id
            ]["dominant_subject_key"]
        )

        secondary_subject_key = (
            cluster_info[
                secondary_cluster_id
            ]["dominant_subject_key"]
        )

        known_subjects = (
            article_subject_sets[
                article_index
            ]
        )

        if known_subjects:
            labeled_noise_count += 1

            top1_match = (
                bool(
                    primary_subject_key
                )
                and (
                    primary_subject_key
                    in known_subjects
                )
            )

            top2_match = (
                top1_match
                or (
                    bool(
                        secondary_subject_key
                    )
                    and (
                        secondary_subject_key
                        in known_subjects
                    )
                )
            )

            if top1_match:
                top1_match_count += 1

            if top2_match:
                top2_match_count += 1
        else:
            top1_match = None
            top2_match = None

        detail_rows.append(
            {
                "method_name": method_name,
                "method_display_name": (
                    METHOD_DISPLAY_NAMES[
                        method_name
                    ]
                ),
                "row_index": article_index,
                "article_id": articles[
                    article_index
                ].get(
                    "article_id",
                    "",
                ),
                "title_tr": articles[
                    article_index
                ].get(
                    "title_tr",
                    "",
                ),
                "primary_cluster": (
                    primary_cluster_id
                ),
                "primary_topic": (
                    cluster_info[
                        primary_cluster_id
                    ][
                        "dominant_subject_name"
                    ]
                ),
                "primary_score": (
                    primary_score
                ),
                "secondary_cluster": (
                    secondary_cluster_id
                ),
                "secondary_topic": (
                    cluster_info[
                        secondary_cluster_id
                    ][
                        "dominant_subject_name"
                    ]
                ),
                "secondary_score": (
                    secondary_score
                ),
                "score_margin": (
                    score_margin
                ),
                "known_subjects": " | ".join(
                    sorted(
                        subject_display_names.get(
                            subject_key,
                            subject_key,
                        )
                        for subject_key
                        in known_subjects
                    )
                ),
                "top1_matches_metadata": (
                    top1_match
                ),
                "top2_matches_metadata": (
                    top2_match
                ),
            }
        )

    margin_array = np.array(
        margins,
        dtype=np.float32,
    )

    summary = {
        "method_name": method_name,
        "method_display_name": (
            METHOD_DISPLAY_NAMES[
                method_name
            ]
        ),
        "noise_article_count": len(
            noise_indices
        ),
        "subject_labeled_noise_count": (
            labeled_noise_count
        ),
        "noise_top1_metadata_consistency": (
            top1_match_count
            / labeled_noise_count
            if labeled_noise_count
            else 0.0
        ),
        "noise_top2_metadata_consistency": (
            top2_match_count
            / labeled_noise_count
            if labeled_noise_count
            else 0.0
        ),
        "h01_direct_cluster_recovery_rate": (
            cluster_recovery_rate
        ),
        "mean_noise_score_margin": float(
            np.mean(margin_array)
        ),
        "median_noise_score_margin": float(
            np.median(margin_array)
        ),
        "noise_margin_q25": float(
            np.percentile(
                margin_array,
                25,
            )
        ),
        "noise_margin_q75": float(
            np.percentile(
                margin_array,
                75,
            )
        ),
    }

    return (
        summary,
        detail_rows,
    )


# =========================================================
# 5. DOSYALAR
# =========================================================


def save_cluster_representation_csv(
    cluster_ids: List[int],
    cluster_info: Dict[int, Dict[str, Any]],
) -> Path:
    """Cluster temsilcilerini ve konu adlarını kaydeder."""

    output_path = (
        get_output_directory()
        / "day24_cluster_representations.csv"
    )

    fieldnames = [
        "cluster_id",
        "cluster_size",
        "dominant_subject",
        "dominant_subject_count",
        "subject_labeled_count",
        "subject_purity",
        "medoid_row_index",
        "medoid_article_id",
        "medoid_title",
        "top5_core_row_indices",
        "top10_core_row_indices",
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

        for cluster_id in cluster_ids:
            info = cluster_info[
                cluster_id
            ]

            writer.writerow(
                {
                    "cluster_id": (
                        cluster_id
                    ),
                    "cluster_size": (
                        info[
                            "cluster_size"
                        ]
                    ),
                    "dominant_subject": (
                        info[
                            "dominant_subject_name"
                        ]
                    ),
                    "dominant_subject_count": (
                        info[
                            "dominant_subject_count"
                        ]
                    ),
                    "subject_labeled_count": (
                        info[
                            "subject_labeled_count"
                        ]
                    ),
                    "subject_purity": (
                        info[
                            "subject_purity"
                        ]
                    ),
                    "medoid_row_index": (
                        info[
                            "medoid_index"
                        ]
                    ),
                    "medoid_article_id": (
                        info[
                            "medoid_article_id"
                        ]
                    ),
                    "medoid_title": (
                        info[
                            "medoid_title"
                        ]
                    ),
                    "top5_core_row_indices": (
                        " | ".join(
                            str(index)
                            for index
                            in info[
                                "core_indices_by_size"
                            ][5]
                        )
                    ),
                    "top10_core_row_indices": (
                        " | ".join(
                            str(index)
                            for index
                            in info[
                                "core_indices_by_size"
                            ][10]
                        )
                    ),
                }
            )

    return output_path


def save_summary_files(
    summaries: List[Dict[str, Any]],
) -> Tuple[Path, Path]:
    """Yöntem özetlerini CSV ve JSON olarak kaydeder."""

    csv_path = (
        get_output_directory()
        / "day24_noise_assignment_method_summary.csv"
    )

    json_path = (
        get_output_directory()
        / "day24_noise_assignment_method_summary.json"
    )

    with csv_path.open(
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

    with json_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            summaries,
            json_file,
            ensure_ascii=False,
            indent=2,
        )

    return (
        csv_path,
        json_path,
    )


def save_detail_csv(
    detail_rows: List[Dict[str, Any]],
) -> Path:
    """Bütün noise atama sonuçlarını kaydeder."""

    output_path = (
        get_output_directory()
        / "day24_noise_assignment_method_details.csv"
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


def create_comparison_chart(
    summaries: List[Dict[str, Any]],
) -> Path:
    """Ana kalite metriklerini görselleştirir."""

    output_path = (
        get_output_directory()
        / "day24_noise_assignment_comparison.png"
    )

    method_names = [
        summary[
            "method_display_name"
        ]
        for summary in summaries
    ]

    top1_values = [
        summary[
            "noise_top1_metadata_consistency"
        ] * 100
        for summary in summaries
    ]

    top2_values = [
        summary[
            "noise_top2_metadata_consistency"
        ] * 100
        for summary in summaries
    ]

    recovery_values = [
        summary[
            "h01_direct_cluster_recovery_rate"
        ] * 100
        for summary in summaries
    ]

    x_positions = np.arange(
        len(method_names)
    )

    bar_width = 0.25

    plt.figure(
        figsize=(14, 8)
    )

    plt.bar(
        x_positions - bar_width,
        top1_values,
        width=bar_width,
        label=(
            "Noise Top-1 metadata tutarlılığı"
        ),
    )

    plt.bar(
        x_positions,
        top2_values,
        width=bar_width,
        label=(
            "Noise Top-2 metadata tutarlılığı"
        ),
    )

    plt.bar(
        x_positions + bar_width,
        recovery_values,
        width=bar_width,
        label=(
            "H01 cluster geri-bulma oranı"
        ),
    )

    plt.xticks(
        x_positions,
        method_names,
        rotation=12,
        ha="right",
    )

    plt.ylabel(
        "Oran (%)"
    )

    plt.title(
        "H01 Noise Makaleleri İçin "
        "Atama Yöntemi Karşılaştırması"
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
) -> Path:
    """Okunabilir karşılaştırma raporu oluşturur."""

    output_path = (
        get_output_directory()
        / "day24_noise_assignment_report.md"
    )

    ranked_summaries = sorted(
        summaries,
        key=lambda summary: (
            summary[
                "noise_top1_metadata_consistency"
            ],
            summary[
                "noise_top2_metadata_consistency"
            ],
            summary[
                "h01_direct_cluster_recovery_rate"
            ],
        ),
        reverse=True,
    )

    provisional_best = (
        ranked_summaries[0]
    )

    lines: List[str] = [
        "# H01 Noise Atama Yöntemi Karşılaştırması",
        "",
        (
            "Bu deney yalnızca H01 tarafından noise bırakılan "
            "makalelerin konu çekirdeklerine bağlanma yöntemini "
            "karşılaştırır."
        ),
        "",
        (
            "TR Dizin subject metadata alanları atama girdisi "
            "değildir. Yalnızca sonuçların keşifsel "
            "tutarlılığını ölçmek için kullanılmıştır."
        ),
        "",
        "## Sonuçlar",
        "",
        (
            "| Yöntem | Noise Top-1 | Noise Top-2 | "
            "H01 cluster geri-bulma | Medyan marj |"
        ),
        "|---|---:|---:|---:|---:|",
    ]

    for summary in summaries:
        lines.append(
            f"| {summary['method_display_name']} "
            f"| %{summary['noise_top1_metadata_consistency'] * 100:.2f} "
            f"| %{summary['noise_top2_metadata_consistency'] * 100:.2f} "
            f"| %{summary['h01_direct_cluster_recovery_rate'] * 100:.2f} "
            f"| {summary['median_noise_score_margin']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Geçici en iyi yöntem",
            "",
            (
                f"Metadata Top-1 sıralamasına göre: "
                f"**{provisional_best['method_display_name']}**"
            ),
            "",
            (
                "Bu seçim bağımsız test başarımı değildir. "
                "Top-1 ve Top-2 metadata tutarlılığı ile H01 "
                "cluster geri-bulma oranı birlikte "
                "değerlendirilmelidir."
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
# 6. TERMİNAL
# =========================================================


def print_results(
    summaries: List[Dict[str, Any]],
) -> None:
    """Ana karşılaştırmayı terminalde gösterir."""

    print("\n" + "=" * 120)
    print("NOISE ATAMA YÖNTEMİ KARŞILAŞTIRMASI")
    print("=" * 120)

    header = (
        f"{'Yöntem':31}"
        f"{'Noise Top-1':>14}"
        f"{'Noise Top-2':>14}"
        f"{'H01 geri-bulma':>17}"
        f"{'Ort. marj':>13}"
        f"{'Medyan marj':>15}"
    )

    print("\n" + header)
    print("-" * len(header))

    for summary in summaries:
        print(
            f"{summary['method_display_name']:31}"
            f"{summary['noise_top1_metadata_consistency'] * 100:>13.2f}%"
            f"{summary['noise_top2_metadata_consistency'] * 100:>13.2f}%"
            f"{summary['h01_direct_cluster_recovery_rate'] * 100:>16.2f}%"
            f"{summary['mean_noise_score_margin']:>13.4f}"
            f"{summary['median_noise_score_margin']:>15.4f}"
        )


# =========================================================
# 7. MAIN
# =========================================================


def main() -> None:
    print("=" * 85)
    print("DAY 24 — H01 NOISE ATAMA YÖNTEMLERİ")
    print("=" * 85)

    articles = load_articles()

    embeddings = load_embeddings()

    assignments = (
        load_h01_assignments()
    )

    validate_alignment(
        articles=articles,
        assignments=assignments,
    )

    (
        article_subject_sets,
        subject_display_names,
    ) = build_subject_information(
        articles
    )

    (
        cluster_ids,
        cluster_info,
        representations,
    ) = build_cluster_representations(
        articles=articles,
        embeddings=embeddings,
        assignments=assignments,
        article_subject_sets=(
            article_subject_sets
        ),
        subject_display_names=(
            subject_display_names
        ),
    )

    summaries: List[
        Dict[str, Any]
    ] = []

    all_detail_rows: List[
        Dict[str, Any]
    ] = []

    for method_name, representation_matrix in (
        representations.items()
    ):
        print(
            f"\nÇalışıyor: "
            f"{METHOD_DISPLAY_NAMES[method_name]}"
        )

        (
            summary,
            detail_rows,
        ) = evaluate_assignment_method(
            method_name=method_name,
            representation_matrix=(
                representation_matrix
            ),
            cluster_ids=cluster_ids,
            cluster_info=cluster_info,
            articles=articles,
            embeddings=embeddings,
            assignments=assignments,
            article_subject_sets=(
                article_subject_sets
            ),
            subject_display_names=(
                subject_display_names
            ),
        )

        summaries.append(
            summary
        )

        all_detail_rows.extend(
            detail_rows
        )

    print_results(
        summaries
    )

    cluster_path = (
        save_cluster_representation_csv(
            cluster_ids=cluster_ids,
            cluster_info=cluster_info,
        )
    )

    (
        summary_csv_path,
        summary_json_path,
    ) = save_summary_files(
        summaries
    )

    detail_path = save_detail_csv(
        all_detail_rows
    )

    chart_path = (
        create_comparison_chart(
            summaries
        )
    )

    report_path = (
        save_markdown_report(
            summaries
        )
    )

    print("\n" + "=" * 85)
    print("DAY 24 TAMAMLANDI")
    print("=" * 85)

    print(
        f"\nCluster temsilleri:\n"
        f"{cluster_path}"
    )

    print(
        f"\nYöntem özet CSV:\n"
        f"{summary_csv_path}"
    )

    print(
        f"\nYöntem özet JSON:\n"
        f"{summary_json_path}"
    )

    print(
        f"\nNoise makale detayları:\n"
        f"{detail_path}"
    )

    print(
        f"\nKarşılaştırma görseli:\n"
        f"{chart_path}"
    )

    print(
        f"\nOkunabilir rapor:\n"
        f"{report_path}"
    )


if __name__ == "__main__":
    main()