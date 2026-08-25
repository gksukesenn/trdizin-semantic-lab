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
CLUSTER_COUNT = 30
HDBSCAN_CONFIG_ID = "H16"


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


# =========================================================
# 1. VERİLERİ OKUMA
# =========================================================


def load_articles() -> List[Dict[str, Any]]:
    """1.000 makalelik pilot JSONL dosyasını okur."""

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
    """TR-MTEB embedding matrisini yükler."""

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


def load_kmeans_assignments() -> List[Dict[str, Any]]:
    """Day 15 KMeans k=30 sonuçlarını okur."""

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
            "silhouette",
        }

        available_columns = set(
            reader.fieldnames or []
        )

        missing_columns = (
            required_columns - available_columns
        )

        if missing_columns:
            raise ValueError(
                "KMeans CSV dosyasında eksik sütunlar: "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            rows.append(
                {
                    "row_index": int(row["row_index"]),
                    "article_id": row["article_id"],
                    "cluster_id": int(row["cluster_id"]),
                    "silhouette": float(row["silhouette"]),
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
    """Day 17 dosyasından H16 sonuçlarını okur."""

    input_path = (
        get_project_root()
        / "research" / "outputs"
        / "day17_hdbscan_all_assignments.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"HDBSCAN atama dosyası bulunamadı:\n{input_path}"
        )

    label_column = f"{HDBSCAN_CONFIG_ID}_label"
    probability_column = (
        f"{HDBSCAN_CONFIG_ID}_probability"
    )
    outlier_column = (
        f"{HDBSCAN_CONFIG_ID}_outlier_score"
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
            outlier_column,
        }

        available_columns = set(
            reader.fieldnames or []
        )

        missing_columns = (
            required_columns - available_columns
        )

        if missing_columns:
            raise ValueError(
                "HDBSCAN CSV dosyasında eksik sütunlar: "
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
            f"{ARTICLE_COUNT} HDBSCAN ataması bekleniyordu, "
            f"bulunan: {len(rows)}"
        )

    return rows


def validate_alignment(
    articles: List[Dict[str, Any]],
    kmeans_rows: List[Dict[str, Any]],
    hdbscan_rows: List[Dict[str, Any]],
) -> None:
    """Makale sırasının bütün dosyalarda aynı olduğunu doğrular."""

    for row_index in range(ARTICLE_COUNT):
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
# 2. SUBJECT METADATA
# =========================================================


def parse_subject(
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

    full_name = subject.get("fullName")

    if (
        isinstance(full_name, str)
        and full_name.strip()
    ):
        return "full:" + full_name.strip()

    name = subject.get("name")

    if (
        isinstance(name, str)
        and name.strip()
    ):
        return "name:" + name.strip()

    return None


def get_article_subjects(
    article: Dict[str, Any],
) -> Tuple[Set[str], Dict[str, str]]:
    """Makalenin subject anahtarlarını ve adlarını döndürür."""

    subject_keys: Set[str] = set()
    display_names: Dict[str, str] = {}

    raw_subjects = article.get("subjects")

    if not isinstance(raw_subjects, list):
        return subject_keys, display_names

    for raw_subject in raw_subjects:
        subject = parse_subject(raw_subject)

        if subject is None:
            continue

        subject_key = create_subject_key(
            subject
        )

        if not subject_key:
            continue

        full_name = subject.get("fullName")
        name = subject.get("name")

        if (
            isinstance(full_name, str)
            and full_name.strip()
        ):
            display_name = full_name.strip()
        elif (
            isinstance(name, str)
            and name.strip()
        ):
            display_name = name.strip()
        else:
            display_name = subject_key

        subject_keys.add(subject_key)

        display_names[
            subject_key
        ] = display_name

    return subject_keys, display_names


def build_subject_data(
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
        ) = get_article_subjects(article)

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
# 3. KMEANS KÜMELERİNE GEÇİCİ KONU ADI VERME
# =========================================================


def build_cluster_subject_dictionary(
    kmeans_labels: np.ndarray,
    article_subject_sets: List[Set[str]],
) -> Tuple[
    Dict[int, Counter],
    Dict[int, int],
]:
    """
    Her KMeans clusterındaki subjectleri sayar.

    Subjectler clustering sırasında kullanılmamıştır.
    Burada yalnızca kümelere okunabilir ad vermek için
    sonradan kullanılmaktadır.
    """

    cluster_subject_counts: Dict[
        int,
        Counter,
    ] = {}

    cluster_labeled_counts: Dict[
        int,
        int,
    ] = {}

    for cluster_id in range(
        CLUSTER_COUNT
    ):
        cluster_subject_counts[
            cluster_id
        ] = Counter()

        cluster_labeled_counts[
            cluster_id
        ] = 0

    for article_index, cluster_id in enumerate(
        kmeans_labels
    ):
        subject_set = article_subject_sets[
            article_index
        ]

        if not subject_set:
            continue

        cluster_labeled_counts[
            int(cluster_id)
        ] += 1

        for subject_key in subject_set:
            cluster_subject_counts[
                int(cluster_id)
            ][subject_key] += 1

    return (
        cluster_subject_counts,
        cluster_labeled_counts,
    )


def get_top_cluster_subjects(
    cluster_id: int,
    cluster_subject_counts: Dict[int, Counter],
    subject_display_names: Dict[str, str],
    limit: int = 3,
) -> List[Tuple[str, str, int]]:
    """Clusterın en sık subjectlerini döndürür."""

    results: List[
        Tuple[str, str, int]
    ] = []

    for subject_key, count in (
        cluster_subject_counts[
            cluster_id
        ].most_common(limit)
    ):
        results.append(
            (
                subject_key,
                subject_display_names.get(
                    subject_key,
                    subject_key,
                ),
                int(count),
            )
        )

    return results


# =========================================================
# 4. CLUSTER MERKEZLERİ VE İKİNCİ KONU
# =========================================================


def calculate_cluster_centroids(
    embeddings: np.ndarray,
    kmeans_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    KMeans atamalarından cluster merkezlerini yeniden hesaplar.

    raw_centroids:
    Öklid mesafesi için kullanılır.

    normalized_centroids:
    Cosine similarity göstergesi için kullanılır.
    """

    raw_centroids = np.zeros(
        (
            CLUSTER_COUNT,
            embeddings.shape[1],
        ),
        dtype=np.float32,
    )

    for cluster_id in range(
        CLUSTER_COUNT
    ):
        cluster_vectors = embeddings[
            kmeans_labels == cluster_id
        ]

        if len(cluster_vectors) == 0:
            raise ValueError(
                f"Boş KMeans clusterı bulundu: {cluster_id}"
            )

        raw_centroids[
            cluster_id
        ] = cluster_vectors.mean(
            axis=0
        )

    centroid_norms = np.linalg.norm(
        raw_centroids,
        axis=1,
        keepdims=True,
    )

    safe_norms = np.where(
        centroid_norms == 0,
        1,
        centroid_norms,
    )

    normalized_centroids = (
        raw_centroids / safe_norms
    )

    return (
        raw_centroids,
        normalized_centroids,
    )


def calculate_cluster_distances(
    embeddings: np.ndarray,
    raw_centroids: np.ndarray,
) -> np.ndarray:
    """Her makalenin bütün KMeans merkezlerine uzaklığını hesaplar."""

    embedding_squared_norms = np.sum(
        embeddings * embeddings,
        axis=1,
        keepdims=True,
    )

    centroid_squared_norms = np.sum(
        raw_centroids * raw_centroids,
        axis=1,
    )[None, :]

    distance_squared = (
        embedding_squared_norms
        + centroid_squared_norms
        - 2.0
        * (
            embeddings
            @ raw_centroids.T
        )
    )

    distance_squared = np.maximum(
        distance_squared,
        0.0,
    )

    return np.sqrt(
        distance_squared
    )


def find_secondary_cluster(
    distance_row: np.ndarray,
    primary_cluster_id: int,
) -> int:
    """Birincil cluster dışındaki en yakın clusterı bulur."""

    ranked_clusters = np.argsort(
        distance_row
    )

    for cluster_id in ranked_clusters:
        if int(cluster_id) != primary_cluster_id:
            return int(cluster_id)

    raise RuntimeError(
        "İkinci cluster bulunamadı."
    )


# =========================================================
# 5. FİNAL HYBRID ÇIKTI
# =========================================================


def classify_confidence(
    hdbscan_label: int,
    hdbscan_probability: float,
    distance_margin: float,
    probability_median: float,
    margin_median: float,
) -> str:
    """
    Pilot veri içindeki göreli güven seviyesini belirler.

    Bu değer kalibre edilmiş olasılık değildir.
    Yalnızca pilot veri içindeki göreli bir seviyedir.
    """

    if (
        hdbscan_label >= 0
        and hdbscan_probability
        >= probability_median
        and distance_margin
        >= margin_median
    ):
        return "yüksek"

    if (
        hdbscan_label == -1
        and distance_margin
        < margin_median
    ):
        return "düşük"

    return "orta"


def classify_topic_structure(
    hdbscan_label: int,
    distance_margin: float,
    margin_q25: float,
    margin_q75: float,
) -> str:
    """Makalenin tek konuya mı yoksa geçiş alanına mı yakın olduğunu yorumlar."""

    if (
        hdbscan_label >= 0
        and distance_margin
        >= margin_q75
    ):
        return "belirgin konu çekirdeği"

    if distance_margin <= margin_q25:
        return "çok alanlı / geçiş bölgesi adayı"

    if hdbscan_label == -1:
        return "yoğun kümeye bağlanamayan konu adayı"

    return "orta düzeyde ayrışmış konu"


def choose_secondary_topic(
    primary_subject_key: str,
    secondary_subjects: List[
        Tuple[str, str, int]
    ],
    primary_cluster_subjects: List[
        Tuple[str, str, int]
    ],
) -> Tuple[str, str]:
    """Birincil konudan farklı ikinci konu adayını seçer."""

    for subject_key, subject_name, _ in (
        secondary_subjects
    ):
        if subject_key != primary_subject_key:
            return subject_key, subject_name

    for subject_key, subject_name, _ in (
        primary_cluster_subjects[1:]
    ):
        if subject_key != primary_subject_key:
            return subject_key, subject_name

    return "", "İkinci konu belirlenemedi"


def build_final_rows(
    articles: List[Dict[str, Any]],
    embeddings: np.ndarray,
    kmeans_rows: List[Dict[str, Any]],
    hdbscan_rows: List[Dict[str, Any]],
    article_subject_sets: List[Set[str]],
    subject_display_names: Dict[str, str],
    cluster_subject_counts: Dict[int, Counter],
    cluster_labeled_counts: Dict[int, int],
    raw_centroids: np.ndarray,
    normalized_centroids: np.ndarray,
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    """Her makale için final konu önerisini oluşturur."""

    kmeans_labels = np.array(
        [
            row["cluster_id"]
            for row in kmeans_rows
        ],
        dtype=np.int32,
    )

    distances = calculate_cluster_distances(
        embeddings=embeddings,
        raw_centroids=raw_centroids,
    )

    cosine_scores = (
        embeddings
        @ normalized_centroids.T
    )

    margins: List[float] = []

    secondary_clusters: List[int] = []

    nearest_assignment_mismatch_count = 0

    for article_index in range(
        ARTICLE_COUNT
    ):
        primary_cluster_id = int(
            kmeans_labels[
                article_index
            ]
        )

        nearest_cluster_id = int(
            np.argmin(
                distances[
                    article_index
                ]
            )
        )

        if (
            nearest_cluster_id
            != primary_cluster_id
        ):
            nearest_assignment_mismatch_count += 1

        secondary_cluster_id = (
            find_secondary_cluster(
                distance_row=distances[
                    article_index
                ],
                primary_cluster_id=(
                    primary_cluster_id
                ),
            )
        )

        secondary_clusters.append(
            secondary_cluster_id
        )

        primary_distance = float(
            distances[
                article_index,
                primary_cluster_id,
            ]
        )

        secondary_distance = float(
            distances[
                article_index,
                secondary_cluster_id,
            ]
        )

        margins.append(
            secondary_distance
            - primary_distance
        )

    margin_array = np.array(
        margins,
        dtype=np.float32,
    )

    clustered_probabilities = np.array(
        [
            row["probability"]
            for row in hdbscan_rows
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

    probability_median = float(
        np.median(
            clustered_probabilities
        )
    )

    final_rows: List[
        Dict[str, Any]
    ] = []

    top1_match_count = 0
    top2_match_count = 0
    labeled_article_count = 0

    hdbscan_clustered_labeled_count = 0
    hdbscan_clustered_top1_match = 0
    hdbscan_clustered_top2_match = 0

    hdbscan_noise_labeled_count = 0
    hdbscan_noise_top1_match = 0
    hdbscan_noise_top2_match = 0

    confidence_counter = Counter()
    structure_counter = Counter()

    for article_index, article in enumerate(
        articles
    ):
        primary_cluster_id = int(
            kmeans_labels[
                article_index
            ]
        )

        secondary_cluster_id = (
            secondary_clusters[
                article_index
            ]
        )

        primary_subjects = (
            get_top_cluster_subjects(
                cluster_id=primary_cluster_id,
                cluster_subject_counts=(
                    cluster_subject_counts
                ),
                subject_display_names=(
                    subject_display_names
                ),
                limit=3,
            )
        )

        secondary_subjects = (
            get_top_cluster_subjects(
                cluster_id=secondary_cluster_id,
                cluster_subject_counts=(
                    cluster_subject_counts
                ),
                subject_display_names=(
                    subject_display_names
                ),
                limit=3,
            )
        )

        if primary_subjects:
            (
                primary_subject_key,
                primary_topic,
                primary_subject_count,
            ) = primary_subjects[0]
        else:
            primary_subject_key = ""
            primary_topic = (
                f"KMeans Cluster "
                f"{primary_cluster_id}"
            )
            primary_subject_count = 0

        (
            secondary_subject_key,
            secondary_topic,
        ) = choose_secondary_topic(
            primary_subject_key=(
                primary_subject_key
            ),
            secondary_subjects=(
                secondary_subjects
            ),
            primary_cluster_subjects=(
                primary_subjects
            ),
        )

        labeled_cluster_count = (
            cluster_labeled_counts[
                primary_cluster_id
            ]
        )

        cluster_topic_purity = (
            primary_subject_count
            / labeled_cluster_count
            if labeled_cluster_count
            else 0.0
        )

        hdbscan_label = hdbscan_rows[
            article_index
        ]["label"]

        hdbscan_probability = (
            hdbscan_rows[
                article_index
            ]["probability"]
        )

        hdbscan_outlier_score = (
            hdbscan_rows[
                article_index
            ]["outlier_score"]
        )

        distance_margin = float(
            margin_array[
                article_index
            ]
        )

        confidence_level = (
            classify_confidence(
                hdbscan_label=hdbscan_label,
                hdbscan_probability=(
                    hdbscan_probability
                ),
                distance_margin=(
                    distance_margin
                ),
                probability_median=(
                    probability_median
                ),
                margin_median=(
                    margin_median
                ),
            )
        )

        topic_structure = (
            classify_topic_structure(
                hdbscan_label=hdbscan_label,
                distance_margin=(
                    distance_margin
                ),
                margin_q25=margin_q25,
                margin_q75=margin_q75,
            )
        )

        confidence_counter[
            confidence_level
        ] += 1

        structure_counter[
            topic_structure
        ] += 1

        real_subjects = (
            article_subject_sets[
                article_index
            ]
        )

        if real_subjects:
            labeled_article_count += 1

            top1_match = (
                primary_subject_key
                in real_subjects
            )

            top2_match = (
                top1_match
                or (
                    secondary_subject_key
                    in real_subjects
                )
            )

            if top1_match:
                top1_match_count += 1

            if top2_match:
                top2_match_count += 1

            if hdbscan_label >= 0:
                hdbscan_clustered_labeled_count += 1

                if top1_match:
                    hdbscan_clustered_top1_match += 1

                if top2_match:
                    hdbscan_clustered_top2_match += 1
            else:
                hdbscan_noise_labeled_count += 1

                if top1_match:
                    hdbscan_noise_top1_match += 1

                if top2_match:
                    hdbscan_noise_top2_match += 1
        else:
            top1_match = None
            top2_match = None

        final_rows.append(
            {
                "row_index": article_index,
                "article_id": article.get(
                    "article_id",
                    "",
                ),
                "publication_year": article.get(
                    "publication_year",
                    "",
                ),
                "title_tr": article.get(
                    "title_tr",
                    "",
                ),
                "kmeans_primary_cluster": (
                    primary_cluster_id
                ),
                "primary_topic": (
                    primary_topic
                ),
                "primary_cluster_subject_purity": (
                    cluster_topic_purity
                ),
                "secondary_cluster": (
                    secondary_cluster_id
                ),
                "secondary_topic": (
                    secondary_topic
                ),
                "primary_cosine_similarity": float(
                    cosine_scores[
                        article_index,
                        primary_cluster_id,
                    ]
                ),
                "secondary_cosine_similarity": float(
                    cosine_scores[
                        article_index,
                        secondary_cluster_id,
                    ]
                ),
                "primary_distance": float(
                    distances[
                        article_index,
                        primary_cluster_id,
                    ]
                ),
                "secondary_distance": float(
                    distances[
                        article_index,
                        secondary_cluster_id,
                    ]
                ),
                "distance_margin": (
                    distance_margin
                ),
                "kmeans_silhouette": (
                    kmeans_rows[
                        article_index
                    ]["silhouette"]
                ),
                "hdbscan_status": (
                    "clusterlandı"
                    if hdbscan_label >= 0
                    else "noise / belirsiz"
                ),
                "hdbscan_cluster": (
                    hdbscan_label
                ),
                "hdbscan_probability": (
                    hdbscan_probability
                ),
                "hdbscan_outlier_score": (
                    hdbscan_outlier_score
                ),
                "relative_confidence": (
                    confidence_level
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
                        in real_subjects
                    )
                ),
                "primary_topic_matches_known_subject": (
                    top1_match
                ),
                "top2_topics_match_known_subject": (
                    top2_match
                ),
            }
        )

    summary = {
        "article_count": ARTICLE_COUNT,
        "embedding_model": (
            "trmteb/"
            "turkish-embedding-model-fine-tuned"
        ),
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "primary_method": (
            "KMeans k=30"
        ),
        "uncertainty_method": (
            "HDBSCAN H16"
        ),
        "output_coverage": 1.0,
        "hdbscan_clustered_count": sum(
            row["label"] >= 0
            for row in hdbscan_rows
        ),
        "hdbscan_noise_count": sum(
            row["label"] == -1
            for row in hdbscan_rows
        ),
        "subject_labeled_article_count": (
            labeled_article_count
        ),
        "top1_subject_match_rate": (
            top1_match_count
            / labeled_article_count
            if labeled_article_count
            else 0.0
        ),
        "top2_subject_match_rate": (
            top2_match_count
            / labeled_article_count
            if labeled_article_count
            else 0.0
        ),
        "hdbscan_clustered_top1_match_rate": (
            hdbscan_clustered_top1_match
            / hdbscan_clustered_labeled_count
            if hdbscan_clustered_labeled_count
            else 0.0
        ),
        "hdbscan_clustered_top2_match_rate": (
            hdbscan_clustered_top2_match
            / hdbscan_clustered_labeled_count
            if hdbscan_clustered_labeled_count
            else 0.0
        ),
        "hdbscan_noise_top1_match_rate": (
            hdbscan_noise_top1_match
            / hdbscan_noise_labeled_count
            if hdbscan_noise_labeled_count
            else 0.0
        ),
        "hdbscan_noise_top2_match_rate": (
            hdbscan_noise_top2_match
            / hdbscan_noise_labeled_count
            if hdbscan_noise_labeled_count
            else 0.0
        ),
        "margin_q25": margin_q25,
        "margin_median": margin_median,
        "margin_q75": margin_q75,
        "hdbscan_probability_median": (
            probability_median
        ),
        "confidence_distribution": dict(
            confidence_counter
        ),
        "topic_structure_distribution": dict(
            structure_counter
        ),
        "nearest_centroid_assignment_mismatch_count": (
            nearest_assignment_mismatch_count
        ),
        "important_note": (
            "Konu adları KMeans clusterlarının baskın "
            "TR Dizin subjectlerinden türetilen geçici "
            "etiketlerdir. Subjectler embedding veya "
            "clustering girdisi değildir. Güven seviyeleri "
            "kalibre edilmiş olasılık değil, pilot veri "
            "içindeki göreli seviyelerdir."
        ),
    }

    return final_rows, summary


# =========================================================
# 6. DOSYALARI KAYDETME
# =========================================================


def save_cluster_dictionary(
    cluster_subject_counts: Dict[int, Counter],
    cluster_labeled_counts: Dict[int, int],
    subject_display_names: Dict[str, str],
) -> Path:
    """30 KMeans clusterının geçici konu sözlüğünü kaydeder."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day21_cluster_topic_dictionary.csv"
    )

    fieldnames = [
        "cluster_id",
        "subject_labeled_article_count",
        "primary_topic",
        "primary_topic_count",
        "primary_topic_purity",
        "secondary_topic",
        "secondary_topic_count",
        "third_topic",
        "third_topic_count",
        "label_status",
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

        for cluster_id in range(
            CLUSTER_COUNT
        ):
            top_subjects = (
                get_top_cluster_subjects(
                    cluster_id=cluster_id,
                    cluster_subject_counts=(
                        cluster_subject_counts
                    ),
                    subject_display_names=(
                        subject_display_names
                    ),
                    limit=3,
                )
            )

            while len(top_subjects) < 3:
                top_subjects.append(
                    (
                        "",
                        "",
                        0,
                    )
                )

            labeled_count = (
                cluster_labeled_counts[
                    cluster_id
                ]
            )

            primary_count = (
                top_subjects[0][2]
            )

            writer.writerow(
                {
                    "cluster_id": cluster_id,
                    "subject_labeled_article_count": (
                        labeled_count
                    ),
                    "primary_topic": (
                        top_subjects[0][1]
                    ),
                    "primary_topic_count": (
                        primary_count
                    ),
                    "primary_topic_purity": (
                        primary_count
                        / labeled_count
                        if labeled_count
                        else 0.0
                    ),
                    "secondary_topic": (
                        top_subjects[1][1]
                    ),
                    "secondary_topic_count": (
                        top_subjects[1][2]
                    ),
                    "third_topic": (
                        top_subjects[2][1]
                    ),
                    "third_topic_count": (
                        top_subjects[2][2]
                    ),
                    "label_status": (
                        "Metadata-derived provisional label"
                    ),
                }
            )

    return output_path


def save_final_assignments(
    final_rows: List[Dict[str, Any]],
) -> Path:
    """Makale bazlı final hybrid sonucu CSV olarak kaydeder."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day21_final_topic_assignments.csv"
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


def save_summary(
    summary: Dict[str, Any],
) -> Path:
    """Pipeline özetini JSON olarak kaydeder."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day21_final_pipeline_summary.json"
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
    """Göreli güven seviyelerini görselleştirir."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day21_confidence_distribution.png"
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
        "Final Hybrid Pipeline — Göreli Güven Dağılımı"
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
    """Final pipeline sonucunu okunabilir Markdown raporuna yazar."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day21_final_pipeline_report.md"
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

    lines = [
        "# Final Pilot Hybrid Topic Pipeline",
        "",
        "## Dondurulan yöntem kararı",
        "",
        "- Embedding: TR-MTEB",
        "- Ana clustering: KMeans k=30",
        "- Belirsizlik kontrolü: HDBSCAN H16",
        "- Çıktı: Birincil konu, ikincil konu ve göreli güven",
        "",
        "## Kapsama",
        "",
        (
            f"- Toplam makale: "
            f"{summary['article_count']}"
        ),
        "- Final konu çıktısı üretilen: 1000 (%100)",
        (
            f"- HDBSCAN çekirdek kümelerine giren: "
            f"{summary['hdbscan_clustered_count']}"
        ),
        (
            f"- HDBSCAN noise/belirsiz: "
            f"{summary['hdbscan_noise_count']}"
        ),
        "",
        "## Subject ile keşifsel uyum",
        "",
        (
            f"- Top-1 subject uyumu: "
            f"%{summary['top1_subject_match_rate'] * 100:.2f}"
        ),
        (
            f"- Top-2 subject uyumu: "
            f"%{summary['top2_subject_match_rate'] * 100:.2f}"
        ),
        (
            f"- HDBSCAN clusterlananlarda Top-2: "
            f"%{summary['hdbscan_clustered_top2_match_rate'] * 100:.2f}"
        ),
        (
            f"- HDBSCAN noise grubunda Top-2: "
            f"%{summary['hdbscan_noise_top2_match_rate'] * 100:.2f}"
        ),
        "",
        (
            "Bu değerler bağımsız test başarımı değildir. "
            "Kümelerin baskın subjectleriyle yapılan keşifsel "
            "uyum kontrolüdür."
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
        "## Konu yapısı",
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
            "## Önemli sınırlamalar",
            "",
            (
                "- Cluster konu adları TR Dizin subject "
                "metadata alanlarından türetilen geçici adlardır."
            ),
            (
                "- Subjectler embedding ve clustering girdisi "
                "olarak kullanılmamıştır."
            ),
            (
                "- Göreli güven seviyesi kalibre edilmiş bir "
                "olasılık değildir."
            ),
            (
                "- Nihai başarı için ayrı eğitim/test veya "
                "insan değerlendirme kümesi gerekir."
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
# 7. MAIN
# =========================================================


def main() -> None:
    print("=" * 85)
    print("DAY 21 — FINAL HYBRID TOPIC PIPELINE")
    print("=" * 85)

    articles = load_articles()
    embeddings = load_embeddings()

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
        subject_display_names,
    ) = build_subject_data(
        articles
    )

    kmeans_labels = np.array(
        [
            row["cluster_id"]
            for row in kmeans_rows
        ],
        dtype=np.int32,
    )

    (
        cluster_subject_counts,
        cluster_labeled_counts,
    ) = build_cluster_subject_dictionary(
        kmeans_labels=kmeans_labels,
        article_subject_sets=(
            article_subject_sets
        ),
    )

    (
        raw_centroids,
        normalized_centroids,
    ) = calculate_cluster_centroids(
        embeddings=embeddings,
        kmeans_labels=kmeans_labels,
    )

    (
        final_rows,
        summary,
    ) = build_final_rows(
        articles=articles,
        embeddings=embeddings,
        kmeans_rows=kmeans_rows,
        hdbscan_rows=hdbscan_rows,
        article_subject_sets=(
            article_subject_sets
        ),
        subject_display_names=(
            subject_display_names
        ),
        cluster_subject_counts=(
            cluster_subject_counts
        ),
        cluster_labeled_counts=(
            cluster_labeled_counts
        ),
        raw_centroids=raw_centroids,
        normalized_centroids=(
            normalized_centroids
        ),
    )

    dictionary_path = (
        save_cluster_dictionary(
            cluster_subject_counts=(
                cluster_subject_counts
            ),
            cluster_labeled_counts=(
                cluster_labeled_counts
            ),
            subject_display_names=(
                subject_display_names
            ),
        )
    )

    assignments_path = (
        save_final_assignments(
            final_rows
        )
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
    print("FINAL PİLOT SONUÇ")
    print("=" * 85)

    print(
        f"\nToplam makale          : "
        f"{summary['article_count']}"
    )

    print(
        "Konu çıktısı üretilen : "
        "1000 (%100)"
    )

    print(
        f"HDBSCAN çekirdek       : "
        f"{summary['hdbscan_clustered_count']}"
    )

    print(
        f"HDBSCAN belirsiz/noise : "
        f"{summary['hdbscan_noise_count']}"
    )

    print(
        f"\nTop-1 subject uyumu    : "
        f"%{summary['top1_subject_match_rate'] * 100:.2f}"
    )

    print(
        f"Top-2 subject uyumu    : "
        f"%{summary['top2_subject_match_rate'] * 100:.2f}"
    )

    print(
        "\nNot: Bu uyum oranları bağımsız test başarımı "
        "değildir; keşifsel metadata kontrolüdür."
    )

    print("\n" + "=" * 85)
    print("DOSYALAR")
    print("=" * 85)

    print(
        f"\nCluster konu sözlüğü:\n"
        f"{dictionary_path}"
    )

    print(
        f"\nMakale bazlı final çıktılar:\n"
        f"{assignments_path}"
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
        f"\nOkunabilir final rapor:\n"
        f"{report_path}"
    )


if __name__ == "__main__":
    main()