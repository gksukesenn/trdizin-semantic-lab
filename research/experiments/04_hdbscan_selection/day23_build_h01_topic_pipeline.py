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

TOP_SUBJECT_COUNT = 3
TOP_KEYWORD_COUNT = 10


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


def load_embeddings() -> np.ndarray:
    """TR-MTEB embeddinglerini yükler ve normalize eder."""

    input_path = (
        get_project_root()
        / "research" / "outputs"
        / "day13_embeddings"
        / "tr_mteb.npy"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Embedding dosyası bulunamadı:\n{input_path}"
        )

    embeddings = np.load(input_path)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embedding matrisi iki boyutlu değil: "
            f"{embeddings.shape}"
        )

    if embeddings.shape[0] != ARTICLE_COUNT:
        raise ValueError(
            f"{ARTICLE_COUNT} embedding bekleniyordu, "
            f"bulunan: {embeddings.shape[0]}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embedding matrisinde NaN veya sonsuz değer var."
        )

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


def load_h01_assignments() -> List[Dict[str, Any]]:
    """Day 17 dosyasından H01 atamalarını okur."""

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
                        row[probability_column]
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
            f"{ARTICLE_COUNT} atama bekleniyordu, "
            f"bulunan: {len(rows)}"
        )

    return rows


def validate_alignment(
    articles: List[Dict[str, Any]],
    assignments: List[Dict[str, Any]],
) -> None:
    """Makale ve atama sırasının aynı olduğunu doğrular."""

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
                f"row_index uyuşmazlığı: {row_index}"
            )

        if (
            assignment["article_id"]
            != article_id
        ):
            raise ValueError(
                f"article_id uyuşmazlığı: {row_index}"
            )


# =========================================================
# 2. METADATA HAZIRLAMA
# =========================================================


def normalize_text(value: Any) -> str:
    """Metni temiz ve tek satırlı hâle getirir."""

    if not isinstance(value, str):
        return ""

    return " ".join(
        value.split()
    ).strip()


def normalize_keywords(
    value: Any,
) -> List[str]:
    """Anahtar kelime alanını temizler."""

    if not isinstance(value, list):
        return []

    keywords: List[str] = []

    for item in value:
        cleaned_item = normalize_text(
            item
        )

        if cleaned_item:
            keywords.append(
                cleaned_item
            )

    return keywords


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
    """Subject için kararlı bir anahtar üretir."""

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


def get_article_subjects(
    article: Dict[str, Any],
) -> Tuple[
    Set[str],
    Dict[str, str],
]:
    """Makalenin subject anahtarlarını ve adlarını döndürür."""

    subject_keys: Set[str] = set()

    display_names: Dict[
        str,
        str,
    ] = {}

    raw_subjects = article.get(
        "subjects"
    )

    if not isinstance(
        raw_subjects,
        list,
    ):
        return (
            subject_keys,
            display_names,
        )

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

        display_names[
            subject_key
        ] = display_name

    return (
        subject_keys,
        display_names,
    )


def build_subject_information(
    articles: List[Dict[str, Any]],
) -> Tuple[
    List[Set[str]],
    Dict[str, str],
]:
    """Bütün makalelerin subject yapılarını hazırlar."""

    article_subject_sets: List[
        Set[str]
    ] = []

    subject_display_names: Dict[
        str,
        str,
    ] = {}

    for article in articles:
        (
            subject_keys,
            display_names,
        ) = get_article_subjects(
            article
        )

        article_subject_sets.append(
            subject_keys
        )

        subject_display_names.update(
            display_names
        )

    return (
        article_subject_sets,
        subject_display_names,
    )


# =========================================================
# 3. H01 CLUSTER ÇEKİRDEKLERİ VE MEDOİDLER
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


def find_cluster_medoid(
    cluster_indices: List[int],
    embeddings: np.ndarray,
) -> Tuple[int, float]:
    """
    Cluster içindeki gerçek medoid makaleyi bulur.

    Medoid:
    Aynı clusterdaki diğer makalelere ortalama cosine
    benzerliği en yüksek olan gerçek makaledir.
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

    medoid_article_index = int(
        cluster_indices[
            best_local_index
        ]
    )

    medoid_mean_similarity = float(
        mean_similarities[
            best_local_index
        ]
    )

    return (
        medoid_article_index,
        medoid_mean_similarity,
    )


def count_cluster_subjects(
    cluster_indices: List[int],
    article_subject_sets: List[Set[str]],
) -> Counter:
    """Cluster içindeki subjectlerin makale sıklığını hesaplar."""

    subject_counter = Counter()

    for article_index in cluster_indices:
        for subject_key in (
            article_subject_sets[
                article_index
            ]
        ):
            subject_counter[
                subject_key
            ] += 1

    return subject_counter


def count_cluster_keywords(
    cluster_indices: List[int],
    articles: List[Dict[str, Any]],
) -> List[Tuple[str, int]]:
    """Cluster içindeki anahtar kelimeleri sayar."""

    keyword_counter = Counter()

    display_names: Dict[
        str,
        str,
    ] = {}

    for article_index in cluster_indices:
        article_keywords = (
            normalize_keywords(
                articles[
                    article_index
                ].get(
                    "keywords_tr"
                )
            )
        )

        unique_keywords = set()

        for keyword in article_keywords:
            normalized_keyword = (
                keyword
                .casefold()
                .strip(
                    " .,:;"
                )
            )

            if not normalized_keyword:
                continue

            unique_keywords.add(
                normalized_keyword
            )

            display_names.setdefault(
                normalized_keyword,
                keyword,
            )

        for normalized_keyword in (
            unique_keywords
        ):
            keyword_counter[
                normalized_keyword
            ] += 1

    return [
        (
            display_names[
                normalized_keyword
            ],
            count,
        )
        for normalized_keyword, count
        in keyword_counter.most_common(
            TOP_KEYWORD_COUNT
        )
    ]


def build_cluster_dictionary(
    articles: List[Dict[str, Any]],
    embeddings: np.ndarray,
    assignments: List[Dict[str, Any]],
    article_subject_sets: List[Set[str]],
    subject_display_names: Dict[str, str],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[int, Dict[str, Any]],
    np.ndarray,
]:
    """
    Her H01 clusterı için:
    - medoid,
    - baskın subject,
    - keywords,
    - otomatik geçici konu etiketi

    oluşturur.
    """

    cluster_ids = get_cluster_ids(
        assignments
    )

    dictionary_rows: List[
        Dict[str, Any]
    ] = []

    cluster_info: Dict[
        int,
        Dict[str, Any],
    ] = {}

    medoid_vectors: List[
        np.ndarray
    ] = []

    for cluster_id in cluster_ids:
        cluster_indices = [
            row_index
            for row_index, assignment
            in enumerate(assignments)
            if assignment["label"]
            == cluster_id
        ]

        (
            medoid_index,
            medoid_mean_similarity,
        ) = find_cluster_medoid(
            cluster_indices=(
                cluster_indices
            ),
            embeddings=embeddings,
        )

        medoid_vectors.append(
            embeddings[
                medoid_index
            ]
        )

        subject_counter = (
            count_cluster_subjects(
                cluster_indices=(
                    cluster_indices
                ),
                article_subject_sets=(
                    article_subject_sets
                ),
            )
        )

        labeled_article_count = sum(
            bool(
                article_subject_sets[
                    article_index
                ]
            )
            for article_index
            in cluster_indices
        )

        top_subjects = (
            subject_counter.most_common(
                TOP_SUBJECT_COUNT
            )
        )

        top_keywords = (
            count_cluster_keywords(
                cluster_indices=(
                    cluster_indices
                ),
                articles=articles,
            )
        )

        if top_subjects:
            primary_subject_key = (
                top_subjects[0][0]
            )

            provisional_topic_label = (
                subject_display_names.get(
                    primary_subject_key,
                    primary_subject_key,
                )
            )

            label_source = (
                "baskın TR Dizin subject metadata"
            )

            primary_subject_count = int(
                top_subjects[0][1]
            )
        elif top_keywords:
            primary_subject_key = ""

            provisional_topic_label = (
                top_keywords[0][0]
            )

            label_source = (
                "keyword fallback"
            )

            primary_subject_count = 0
        else:
            primary_subject_key = ""

            provisional_topic_label = (
                f"H01 Cluster {cluster_id}"
            )

            label_source = (
                "cluster ID fallback"
            )

            primary_subject_count = 0

        subject_purity = (
            primary_subject_count
            / labeled_article_count
            if labeled_article_count
            else 0.0
        )

        top_subject_text = " | ".join(
            (
                f"{subject_display_names.get(subject_key, subject_key)} "
                f"({count})"
            )
            for subject_key, count
            in top_subjects
        )

        top_keyword_text = " | ".join(
            f"{keyword} ({count})"
            for keyword, count
            in top_keywords
        )

        medoid_article = articles[
            medoid_index
        ]

        row = {
            "cluster_id": cluster_id,
            "cluster_size": len(
                cluster_indices
            ),
            "subject_labeled_article_count": (
                labeled_article_count
            ),
            "provisional_topic_label": (
                provisional_topic_label
            ),
            "primary_subject_key": (
                primary_subject_key
            ),
            "topic_label_source": (
                label_source
            ),
            "primary_subject_purity": (
                subject_purity
            ),
            "top_subjects": (
                top_subject_text
            ),
            "top_keywords": (
                top_keyword_text
            ),
            "medoid_row_index": (
                medoid_index
            ),
            "medoid_article_id": (
                medoid_article.get(
                    "article_id",
                    "",
                )
            ),
            "medoid_title": (
                medoid_article.get(
                    "title_tr",
                    "",
                )
            ),
            "medoid_mean_cluster_similarity": (
                medoid_mean_similarity
            ),
        }

        dictionary_rows.append(
            row
        )

        cluster_info[
            cluster_id
        ] = {
            **row,
            "cluster_indices": (
                cluster_indices
            ),
            "top_subject_items": (
                top_subjects
            ),
        }

    medoid_matrix = np.vstack(
        medoid_vectors
    ).astype(
        np.float32,
        copy=False,
    )

    return (
        dictionary_rows,
        cluster_info,
        medoid_matrix,
    )


# =========================================================
# 4. NOISE VE TÜM MAKALELER İÇİN KONU ATAMASI
# =========================================================


def get_secondary_topic(
    primary_cluster_id: int,
    ranked_cluster_ids: List[int],
) -> int:
    """Birincil kümeden farklı ilk kümeyi döndürür."""

    for cluster_id in ranked_cluster_ids:
        if cluster_id != primary_cluster_id:
            return cluster_id

    raise RuntimeError(
        "İkinci konu kümesi bulunamadı."
    )


def classify_relative_confidence(
    original_hdbscan_label: int,
    hdbscan_probability: float,
    similarity_margin: float,
    probability_q25: float,
    probability_median: float,
    margin_q25: float,
    margin_median: float,
) -> str:
    """
    Göreli güven seviyesi üretir.

    Bu değer kalibre edilmiş olasılık değildir.
    Yalnızca pilot veri içindeki göreli bir göstergedir.
    """

    if (
        original_hdbscan_label >= 0
        and hdbscan_probability
        >= probability_median
        and similarity_margin
        >= margin_median
    ):
        return "yüksek"

    if (
        original_hdbscan_label == -1
        and similarity_margin
        <= margin_q25
    ):
        return "düşük"

    if (
        original_hdbscan_label >= 0
        and hdbscan_probability
        < probability_q25
        and similarity_margin
        <= margin_q25
    ):
        return "düşük"

    return "orta"


def classify_topic_structure(
    original_hdbscan_label: int,
    similarity_margin: float,
    margin_q25: float,
    margin_q75: float,
) -> str:
    """Tek konulu veya çok alanlı yapıya ilişkin yorum üretir."""

    if similarity_margin <= margin_q25:
        return (
            "çok alanlı / geçiş bölgesi adayı"
        )

    if (
        original_hdbscan_label >= 0
        and similarity_margin
        >= margin_q75
    ):
        return (
            "belirgin HDBSCAN konu çekirdeği"
        )

    if original_hdbscan_label == -1:
        return (
            "noise iken en yakın konu çekirdeğine bağlandı"
        )

    return (
        "orta düzeyde ayrışmış konu"
    )


def build_final_assignments(
    articles: List[Dict[str, Any]],
    embeddings: np.ndarray,
    assignments: List[Dict[str, Any]],
    article_subject_sets: List[Set[str]],
    subject_display_names: Dict[str, str],
    cluster_info: Dict[int, Dict[str, Any]],
    medoid_matrix: np.ndarray,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    """Her makale için final H01 tabanlı konu çıktısını oluşturur."""

    cluster_ids = sorted(
        cluster_info.keys()
    )

    cluster_position = {
        cluster_id: position
        for position, cluster_id
        in enumerate(cluster_ids)
    }

    similarity_matrix = (
        embeddings
        @ medoid_matrix.T
    )

    raw_rows: List[
        Dict[str, Any]
    ] = []

    margins: List[float] = []

    direct_nearest_medoid_mismatch_count = 0

    for row_index in range(
        ARTICLE_COUNT
    ):
        assignment = assignments[
            row_index
        ]

        original_label = int(
            assignment["label"]
        )

        similarity_row = (
            similarity_matrix[
                row_index
            ]
        )

        ranked_positions = np.argsort(
            similarity_row
        )[::-1]

        ranked_cluster_ids = [
            cluster_ids[
                int(position)
            ]
            for position
            in ranked_positions
        ]

        if original_label >= 0:
            primary_cluster_id = (
                original_label
            )

            assignment_method = (
                "doğrudan HDBSCAN H01"
            )

            nearest_medoid_cluster_id = (
                ranked_cluster_ids[0]
            )

            if (
                nearest_medoid_cluster_id
                != original_label
            ):
                direct_nearest_medoid_mismatch_count += 1
        else:
            primary_cluster_id = (
                ranked_cluster_ids[0]
            )

            assignment_method = (
                "noise → en yakın H01 medoid"
            )

        secondary_cluster_id = (
            get_secondary_topic(
                primary_cluster_id=(
                    primary_cluster_id
                ),
                ranked_cluster_ids=(
                    ranked_cluster_ids
                ),
            )
        )

        primary_similarity = float(
            similarity_row[
                cluster_position[
                    primary_cluster_id
                ]
            ]
        )

        secondary_similarity = float(
            similarity_row[
                cluster_position[
                    secondary_cluster_id
                ]
            ]
        )

        similarity_margin = (
            primary_similarity
            - secondary_similarity
        )

        margins.append(
            similarity_margin
        )

        raw_rows.append(
            {
                "row_index": row_index,
                "article_id": articles[
                    row_index
                ].get(
                    "article_id",
                    "",
                ),
                "publication_year": articles[
                    row_index
                ].get(
                    "publication_year",
                    "",
                ),
                "title_tr": articles[
                    row_index
                ].get(
                    "title_tr",
                    "",
                ),
                "original_hdbscan_status": (
                    "clusterlandı"
                    if original_label >= 0
                    else "noise / belirsiz"
                ),
                "original_hdbscan_cluster": (
                    original_label
                ),
                "hdbscan_probability": (
                    assignment[
                        "probability"
                    ]
                ),
                "hdbscan_outlier_score": (
                    assignment[
                        "outlier_score"
                    ]
                ),
                "assignment_method": (
                    assignment_method
                ),
                "primary_cluster": (
                    primary_cluster_id
                ),
                "primary_topic": (
                    cluster_info[
                        primary_cluster_id
                    ][
                        "provisional_topic_label"
                    ]
                ),
                "primary_subject_key": (
                    cluster_info[
                        primary_cluster_id
                    ][
                        "primary_subject_key"
                    ]
                ),
                "primary_medoid_article_id": (
                    cluster_info[
                        primary_cluster_id
                    ][
                        "medoid_article_id"
                    ]
                ),
                "primary_medoid_title": (
                    cluster_info[
                        primary_cluster_id
                    ][
                        "medoid_title"
                    ]
                ),
                "primary_medoid_similarity": (
                    primary_similarity
                ),
                "secondary_cluster": (
                    secondary_cluster_id
                ),
                "secondary_topic": (
                    cluster_info[
                        secondary_cluster_id
                    ][
                        "provisional_topic_label"
                    ]
                ),
                "secondary_subject_key": (
                    cluster_info[
                        secondary_cluster_id
                    ][
                        "primary_subject_key"
                    ]
                ),
                "secondary_medoid_similarity": (
                    secondary_similarity
                ),
                "similarity_margin": (
                    similarity_margin
                ),
            }
        )

    margin_array = np.array(
        margins,
        dtype=np.float32,
    )

    clustered_probabilities = np.array(
        [
            row["probability"]
            for row in assignments
            if row["label"] >= 0
        ],
        dtype=np.float32,
    )

    margin_q25 = float(
        np.percentile(
            margin_array,
            25,
        )
    )

    margin_median = float(
        np.median(
            margin_array
        )
    )

    margin_q75 = float(
        np.percentile(
            margin_array,
            75,
        )
    )

    probability_q25 = float(
        np.percentile(
            clustered_probabilities,
            25,
        )
    )

    probability_median = float(
        np.median(
            clustered_probabilities
        )
    )

    final_rows: List[
        Dict[str, Any]
    ] = []

    confidence_counter = Counter()
    structure_counter = Counter()

    subject_labeled_count = 0
    metadata_top1_consistency_count = 0
    metadata_top2_consistency_count = 0

    clustered_subject_labeled_count = 0
    clustered_top1_consistency_count = 0

    noise_subject_labeled_count = 0
    noise_top1_consistency_count = 0

    for raw_row in raw_rows:
        row_index = int(
            raw_row["row_index"]
        )

        original_label = int(
            raw_row[
                "original_hdbscan_cluster"
            ]
        )

        relative_confidence = (
            classify_relative_confidence(
                original_hdbscan_label=(
                    original_label
                ),
                hdbscan_probability=float(
                    raw_row[
                        "hdbscan_probability"
                    ]
                ),
                similarity_margin=float(
                    raw_row[
                        "similarity_margin"
                    ]
                ),
                probability_q25=(
                    probability_q25
                ),
                probability_median=(
                    probability_median
                ),
                margin_q25=margin_q25,
                margin_median=(
                    margin_median
                ),
            )
        )

        topic_structure = (
            classify_topic_structure(
                original_hdbscan_label=(
                    original_label
                ),
                similarity_margin=float(
                    raw_row[
                        "similarity_margin"
                    ]
                ),
                margin_q25=margin_q25,
                margin_q75=margin_q75,
            )
        )

        confidence_counter[
            relative_confidence
        ] += 1

        structure_counter[
            topic_structure
        ] += 1

        known_subjects = (
            article_subject_sets[
                row_index
            ]
        )

        primary_subject_key = str(
            raw_row[
                "primary_subject_key"
            ]
        )

        secondary_subject_key = str(
            raw_row[
                "secondary_subject_key"
            ]
        )

        if known_subjects:
            subject_labeled_count += 1

            top1_consistent = (
                bool(primary_subject_key)
                and primary_subject_key
                in known_subjects
            )

            top2_consistent = (
                top1_consistent
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

            if top1_consistent:
                metadata_top1_consistency_count += 1

            if top2_consistent:
                metadata_top2_consistency_count += 1

            if original_label >= 0:
                clustered_subject_labeled_count += 1

                if top1_consistent:
                    clustered_top1_consistency_count += 1
            else:
                noise_subject_labeled_count += 1

                if top1_consistent:
                    noise_top1_consistency_count += 1
        else:
            top1_consistent = None
            top2_consistent = None

        final_rows.append(
            {
                **raw_row,
                "relative_confidence": (
                    relative_confidence
                ),
                "topic_structure": (
                    topic_structure
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
                "primary_topic_matches_metadata": (
                    top1_consistent
                ),
                "top2_topics_match_metadata": (
                    top2_consistent
                ),
            }
        )

    clustered_count = sum(
        row["label"] >= 0
        for row in assignments
    )

    noise_count = (
        ARTICLE_COUNT
        - clustered_count
    )

    summary = {
        "article_count": (
            ARTICLE_COUNT
        ),
        "embedding_model": (
            "trmteb/"
            "turkish-embedding-model-fine-tuned"
        ),
        "main_clustering_method": (
            "HDBSCAN H01"
        ),
        "hdbscan_parameters": {
            "cluster_selection_method": (
                "eom"
            ),
            "min_cluster_size": 10,
            "min_samples": 5,
            "clustering_input": (
                "TR-MTEB embeddinglerinin "
                "10D UMAP gösterimi"
            ),
        },
        "cluster_count": len(
            cluster_ids
        ),
        "directly_clustered_count": (
            clustered_count
        ),
        "directly_clustered_rate": (
            clustered_count
            / ARTICLE_COUNT
        ),
        "noise_reassigned_count": (
            noise_count
        ),
        "noise_reassigned_rate": (
            noise_count
            / ARTICLE_COUNT
        ),
        "final_output_coverage": 1.0,
        "subject_labeled_article_count": (
            subject_labeled_count
        ),
        "metadata_top1_consistency_rate": (
            metadata_top1_consistency_count
            / subject_labeled_count
            if subject_labeled_count
            else 0.0
        ),
        "metadata_top2_consistency_rate": (
            metadata_top2_consistency_count
            / subject_labeled_count
            if subject_labeled_count
            else 0.0
        ),
        "clustered_metadata_top1_consistency_rate": (
            clustered_top1_consistency_count
            / clustered_subject_labeled_count
            if clustered_subject_labeled_count
            else 0.0
        ),
        "noise_metadata_top1_consistency_rate": (
            noise_top1_consistency_count
            / noise_subject_labeled_count
            if noise_subject_labeled_count
            else 0.0
        ),
        "margin_q25": margin_q25,
        "margin_median": (
            margin_median
        ),
        "margin_q75": margin_q75,
        "hdbscan_probability_q25": (
            probability_q25
        ),
        "hdbscan_probability_median": (
            probability_median
        ),
        "confidence_distribution": dict(
            confidence_counter
        ),
        "topic_structure_distribution": dict(
            structure_counter
        ),
        "direct_cluster_nearest_medoid_mismatch_count": (
            direct_nearest_medoid_mismatch_count
        ),
        "important_note": (
            "Konu adları H01 kümelerindeki baskın TR Dizin "
            "subject metadata alanlarından otomatik türetilen "
            "geçici adlardır. Metadata uyum oranları bağımsız "
            "test başarımı değildir. Göreli güven seviyeleri "
            "kalibre edilmiş olasılık değildir."
        ),
    }

    return (
        final_rows,
        summary,
    )


# =========================================================
# 5. DOSYALARI KAYDETME
# =========================================================


def save_cluster_dictionary(
    dictionary_rows: List[Dict[str, Any]],
) -> Path:
    """H01 cluster konu sözlüğünü kaydeder."""

    output_path = (
        get_output_directory()
        / "day23_h01_cluster_dictionary.csv"
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                dictionary_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            dictionary_rows
        )

    return output_path


def save_final_assignments(
    final_rows: List[Dict[str, Any]],
) -> Path:
    """1.000 makalenin final konu çıktısını kaydeder."""

    output_path = (
        get_output_directory()
        / "day23_h01_final_topic_assignments.csv"
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                final_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            final_rows
        )

    return output_path


def save_noise_review(
    final_rows: List[Dict[str, Any]],
) -> Path:
    """
    Noise iken sonradan bağlanan 230 makaleyi,
    en belirsizden en belirgine sıralar.
    """

    output_path = (
        get_output_directory()
        / "day23_h01_noise_assignments.csv"
    )

    noise_rows = [
        row
        for row in final_rows
        if (
            row[
                "original_hdbscan_status"
            ]
            == "noise / belirsiz"
        )
    ]

    noise_rows.sort(
        key=lambda row: float(
            row["similarity_margin"]
        )
    )

    fieldnames = [
        "row_index",
        "article_id",
        "title_tr",
        "primary_cluster",
        "primary_topic",
        "secondary_cluster",
        "secondary_topic",
        "primary_medoid_similarity",
        "secondary_medoid_similarity",
        "similarity_margin",
        "relative_confidence",
        "topic_structure",
        "known_subjects",
        "primary_topic_matches_metadata",
        "top2_topics_match_metadata",
    ]

    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(
            noise_rows
        )

    return output_path


def save_summary(
    summary: Dict[str, Any],
) -> Path:
    """Teknik özeti JSON olarak kaydeder."""

    output_path = (
        get_output_directory()
        / "day23_h01_pipeline_summary.json"
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            summary,
            json_file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


def create_confidence_chart(
    summary: Dict[str, Any],
) -> Path:
    """Göreli güven dağılımını görselleştirir."""

    output_path = (
        get_output_directory()
        / "day23_h01_confidence_distribution.png"
    )

    confidence_order = [
        "yüksek",
        "orta",
        "düşük",
    ]

    values = [
        summary[
            "confidence_distribution"
        ].get(
            confidence_level,
            0,
        )
        for confidence_level
        in confidence_order
    ]

    plt.figure(
        figsize=(9, 6)
    )

    bars = plt.bar(
        confidence_order,
        values,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 5,
            str(value),
            ha="center",
            va="bottom",
        )

    plt.title(
        "H01 Tabanlı Final Pilot — Göreli Güven Dağılımı"
    )

    plt.xlabel(
        "Göreli güven seviyesi"
    )

    plt.ylabel(
        "Makale sayısı"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=170,
    )

    plt.close()

    return output_path


def save_markdown_report(
    summary: Dict[str, Any],
) -> Path:
    """Okunabilir final rapor oluşturur."""

    output_path = (
        get_output_directory()
        / "day23_h01_pipeline_report.md"
    )

    confidence_distribution = (
        summary[
            "confidence_distribution"
        ]
    )

    structure_distribution = (
        summary[
            "topic_structure_distribution"
        ]
    )

    lines: List[str] = [
        "# HDBSCAN H01 Tabanlı Final Pilot Pipeline",
        "",
        "## Yöntem",
        "",
        "- Girdi: Türkçe abstract",
        "- Embedding: TR-MTEB",
        "- Ana clustering: HDBSCAN H01",
        "- H01 doğrudan clusterlanan makaleler: kendi kümelerinde tutuldu",
        (
            "- H01 noise makaleler: en yakın birinci ve ikinci "
            "cluster medoidine göre otomatik bağlandı"
        ),
        "",
        "## Kapsama",
        "",
        (
            f"- Toplam makale: "
            f"{summary['article_count']}"
        ),
        (
            f"- Doğrudan HDBSCAN clusterlanan: "
            f"{summary['directly_clustered_count']} "
            f"(%{summary['directly_clustered_rate'] * 100:.2f})"
        ),
        (
            f"- Noise iken medoidle bağlanan: "
            f"{summary['noise_reassigned_count']} "
            f"(%{summary['noise_reassigned_rate'] * 100:.2f})"
        ),
        "- Final konu çıktısı üretilen: 1000 (%100)",
        "",
        "## Metadata ile keşifsel tutarlılık",
        "",
        (
            f"- Top-1 metadata tutarlılığı: "
            f"%{summary['metadata_top1_consistency_rate'] * 100:.2f}"
        ),
        (
            f"- Top-2 metadata tutarlılığı: "
            f"%{summary['metadata_top2_consistency_rate'] * 100:.2f}"
        ),
        (
            f"- Doğrudan clusterlananlarda Top-1: "
            f"%{summary['clustered_metadata_top1_consistency_rate'] * 100:.2f}"
        ),
        (
            f"- Noise grubunda Top-1: "
            f"%{summary['noise_metadata_top1_consistency_rate'] * 100:.2f}"
        ),
        "",
        (
            "Bu değerler bağımsız başarı oranı değildir. "
            "Konu adları da aynı metadata alanından türetildiği "
            "için yalnızca keşifsel tutarlılık göstergesidir."
        ),
        "",
        "## Göreli güven dağılımı",
        "",
        (
            f"- Yüksek: "
            f"{confidence_distribution.get('yüksek', 0)}"
        ),
        (
            f"- Orta: "
            f"{confidence_distribution.get('orta', 0)}"
        ),
        (
            f"- Düşük: "
            f"{confidence_distribution.get('düşük', 0)}"
        ),
        "",
        "## Konu yapısı dağılımı",
        "",
    ]

    for structure_name, count in (
        structure_distribution.items()
    ):
        lines.append(
            f"- {structure_name}: {count}"
        )

    lines.extend(
        [
            "",
            "## Sınırlamalar",
            "",
            (
                "- Cluster konu adları otomatik ve geçicidir; "
                "ground truth değildir."
            ),
            (
                "- HDBSCAN noise makalelerin konusuz olduğu "
                "anlamına gelmez."
            ),
            (
                "- Noise ataması en yakın gerçek cluster medoidine "
                "göre yapılmıştır."
            ),
            (
                "- Göreli güven değeri kalibre edilmiş bir "
                "olasılık değildir."
            ),
            (
                "- 50.000 makalelik aşamada parametreler ve konu "
                "sözlüğü yeniden değerlendirilmelidir."
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
    print("DAY 23 — HDBSCAN H01 TABANLI FINAL PİLOT")
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
        dictionary_rows,
        cluster_info,
        medoid_matrix,
    ) = build_cluster_dictionary(
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

    (
        final_rows,
        summary,
    ) = build_final_assignments(
        articles=articles,
        embeddings=embeddings,
        assignments=assignments,
        article_subject_sets=(
            article_subject_sets
        ),
        subject_display_names=(
            subject_display_names
        ),
        cluster_info=cluster_info,
        medoid_matrix=medoid_matrix,
    )

    dictionary_path = (
        save_cluster_dictionary(
            dictionary_rows
        )
    )

    assignments_path = (
        save_final_assignments(
            final_rows
        )
    )

    noise_path = save_noise_review(
        final_rows
    )

    summary_path = save_summary(
        summary
    )

    chart_path = (
        create_confidence_chart(
            summary
        )
    )

    report_path = (
        save_markdown_report(
            summary
        )
    )

    print("\n" + "=" * 85)
    print("FINAL H01 PIPELINE ÖZETİ")
    print("=" * 85)

    print(
        f"\nToplam makale             : "
        f"{summary['article_count']}"
    )

    print(
        f"H01 doğrudan clusterlanan : "
        f"{summary['directly_clustered_count']} "
        f"(%{summary['directly_clustered_rate'] * 100:.2f})"
    )

    print(
        f"Noise iken otomatik atanan: "
        f"{summary['noise_reassigned_count']} "
        f"(%{summary['noise_reassigned_rate'] * 100:.2f})"
    )

    print(
        "\nFinal çıktı üretilen      : "
        "1000 (%100)"
    )

    print(
        f"\nMetadata Top-1 tutarlılığı: "
        f"%{summary['metadata_top1_consistency_rate'] * 100:.2f}"
    )

    print(
        f"Metadata Top-2 tutarlılığı: "
        f"%{summary['metadata_top2_consistency_rate'] * 100:.2f}"
    )

    print(
        "\nNot: Bunlar bağımsız test başarı oranları "
        "değildir."
    )

    print("\n" + "=" * 85)
    print("DOSYALAR")
    print("=" * 85)

    print(
        f"\nH01 cluster konu sözlüğü:\n"
        f"{dictionary_path}"
    )

    print(
        f"\nMakale bazlı final konu çıktıları:\n"
        f"{assignments_path}"
    )

    print(
        f"\nNoise makale atamaları:\n"
        f"{noise_path}"
    )

    print(
        f"\nTeknik özet:\n"
        f"{summary_path}"
    )

    print(
        f"\nGüven dağılımı görseli:\n"
        f"{chart_path}"
    )

    print(
        f"\nOkunabilir rapor:\n"
        f"{report_path}"
    )


if __name__ == "__main__":
    main()