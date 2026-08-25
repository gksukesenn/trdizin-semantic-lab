import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import hdbscan
import numpy as np
import umap
from hdbscan.prediction import approximate_predict
from sklearn.model_selection import train_test_split

from day24_compare_noise_assignment_methods import (
    build_subject_information,
    get_output_directory,
    load_articles,
    load_embeddings,
)


ARTICLE_COUNT = 1000
TEST_RATE = 0.20
RANDOM_SEED = 42

UMAP_COMPONENTS = 10
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0

HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES = 5
HDBSCAN_SELECTION_METHOD = "eom"


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Tek bir vektörü birim uzunluğa getirir."""

    norm = float(np.linalg.norm(vector))

    if norm == 0:
        return vector.astype(
            np.float32,
            copy=False,
        )

    return (
        vector / norm
    ).astype(
        np.float32,
        copy=False,
    )


def create_train_test_split(
    articles: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Her yılın eğitim ve testte benzer oranda bulunması
    için yayın yılına göre stratified split yapar.
    """

    article_indices = np.arange(
        len(articles),
        dtype=np.int32,
    )

    publication_years = np.array(
        [
            str(
                article.get(
                    "publication_year",
                    "unknown",
                )
            )
            for article in articles
        ]
    )

    train_indices, test_indices = train_test_split(
        article_indices,
        test_size=TEST_RATE,
        random_state=RANDOM_SEED,
        stratify=publication_years,
    )

    return (
        np.sort(train_indices),
        np.sort(test_indices),
    )


def train_umap_and_hdbscan(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
) -> Tuple[
    umap.UMAP,
    hdbscan.HDBSCAN,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """UMAP ve HDBSCAN'i yalnızca eğitim verisinde kurar."""

    print("\n" + "=" * 85)
    print("1. EĞİTİM VERİSİNDE UMAP")
    print("=" * 85)

    reducer = umap.UMAP(
        n_components=UMAP_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=RANDOM_SEED,
        low_memory=True,
    )

    train_reduced = reducer.fit_transform(
        train_embeddings
    )

    test_reduced = reducer.transform(
        test_embeddings
    )

    print(
        f"\nEğitim UMAP şekli: "
        f"{train_reduced.shape}"
    )

    print(
        f"Test UMAP şekli   : "
        f"{test_reduced.shape}"
    )

    print("\n" + "=" * 85)
    print("2. EĞİTİM VERİSİNDE HDBSCAN H01")
    print("=" * 85)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=(
            HDBSCAN_MIN_CLUSTER_SIZE
        ),
        min_samples=(
            HDBSCAN_MIN_SAMPLES
        ),
        metric="euclidean",
        cluster_selection_method=(
            HDBSCAN_SELECTION_METHOD
        ),
        prediction_data=True,
        gen_min_span_tree=False,
    )

    train_labels = clusterer.fit_predict(
        train_reduced
    ).astype(
        np.int32,
        copy=False,
    )

    test_labels, test_strengths = approximate_predict(
        clusterer,
        test_reduced,
    )

    test_labels = test_labels.astype(
        np.int32,
        copy=False,
    )

    test_strengths = test_strengths.astype(
        np.float32,
        copy=False,
    )

    cluster_count = len(
        {
            int(label)
            for label in train_labels
            if int(label) >= 0
        }
    )

    train_noise_count = int(
        np.sum(
            train_labels == -1
        )
    )

    print(
        f"\nEğitimde bulunan cluster: "
        f"{cluster_count}"
    )

    print(
        f"Eğitim noise             : "
        f"{train_noise_count}/"
        f"{len(train_labels)} "
        f"(%{train_noise_count / len(train_labels) * 100:.2f})"
    )

    return (
        reducer,
        clusterer,
        train_labels,
        test_labels,
        test_strengths,
    )


def build_training_cluster_dictionary(
    train_indices: np.ndarray,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    article_subject_sets: List[Set[str]],
    subject_display_names: Dict[str, str],
) -> Tuple[
    List[int],
    Dict[int, Dict[str, Any]],
    np.ndarray,
]:
    """
    Cluster adlarını ve centroidleri yalnızca
    eğitim makalelerinden oluşturur.
    """

    cluster_ids = sorted(
        {
            int(label)
            for label in train_labels
            if int(label) >= 0
        }
    )

    if not cluster_ids:
        raise ValueError(
            "Eğitim verisinde HDBSCAN cluster oluşturamadı."
        )

    cluster_info: Dict[
        int,
        Dict[str, Any],
    ] = {}

    centroid_vectors: List[
        np.ndarray
    ] = []

    for cluster_id in cluster_ids:
        local_cluster_indices = np.where(
            train_labels == cluster_id
        )[0]

        global_cluster_indices = [
            int(
                train_indices[
                    local_index
                ]
            )
            for local_index
            in local_cluster_indices
        ]

        cluster_vectors = train_embeddings[
            local_cluster_indices
        ]

        centroid = normalize_vector(
            cluster_vectors.mean(
                axis=0
            )
        )

        centroid_vectors.append(
            centroid
        )

        subject_counter = Counter()

        subject_labeled_count = 0

        for global_index in (
            global_cluster_indices
        ):
            subject_set = (
                article_subject_sets[
                    global_index
                ]
            )

            if not subject_set:
                continue

            subject_labeled_count += 1

            for subject_key in subject_set:
                subject_counter[
                    subject_key
                ] += 1

        if subject_counter:
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
                / subject_labeled_count
                if subject_labeled_count
                else 0.0
            )
        else:
            dominant_subject_key = ""
            dominant_subject_count = 0
            dominant_subject_name = (
                f"Holdout Cluster {cluster_id}"
            )
            subject_purity = 0.0

        cluster_info[
            cluster_id
        ] = {
            "cluster_id": cluster_id,
            "cluster_size": len(
                local_cluster_indices
            ),
            "dominant_subject_key": (
                dominant_subject_key
            ),
            "dominant_subject_name": (
                dominant_subject_name
            ),
            "dominant_subject_count": int(
                dominant_subject_count
            ),
            "subject_labeled_count": (
                subject_labeled_count
            ),
            "subject_purity": float(
                subject_purity
            ),
        }

    centroid_matrix = np.vstack(
        centroid_vectors
    ).astype(
        np.float32,
        copy=False,
    )

    return (
        cluster_ids,
        cluster_info,
        centroid_matrix,
    )


def rank_clusters(
    score_row: np.ndarray,
    cluster_ids: List[int],
) -> List[int]:
    """Centroid skorlarına göre clusterları sıralar."""

    ranked_positions = np.argsort(
        score_row
    )[::-1]

    return [
        cluster_ids[
            int(position)
        ]
        for position in ranked_positions
    ]


def select_secondary_cluster(
    primary_cluster: int,
    ranked_cluster_ids: List[int],
) -> int:
    """Birincil clusterdan farklı ilk clusterı seçer."""

    for cluster_id in ranked_cluster_ids:
        if cluster_id != primary_cluster:
            return cluster_id

    raise RuntimeError(
        "İkinci cluster bulunamadı."
    )


def evaluate_holdout(
    articles: List[Dict[str, Any]],
    test_indices: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    test_strengths: np.ndarray,
    cluster_ids: List[int],
    cluster_info: Dict[int, Dict[str, Any]],
    centroid_matrix: np.ndarray,
    article_subject_sets: List[Set[str]],
    subject_display_names: Dict[str, str],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    """200 holdout makalesi üzerinde pipeline'ı değerlendirir."""

    centroid_scores = (
        test_embeddings
        @ centroid_matrix.T
    )

    cluster_position = {
        cluster_id: position
        for position, cluster_id
        in enumerate(cluster_ids)
    }

    result_rows: List[
        Dict[str, Any]
    ] = []

    direct_count = 0
    fallback_count = 0

    labeled_test_count = 0
    total_top1_count = 0
    total_top2_count = 0

    direct_labeled_count = 0
    direct_top1_count = 0
    direct_top2_count = 0

    fallback_labeled_count = 0
    fallback_top1_count = 0
    fallback_top2_count = 0

    for test_position, global_index in enumerate(
        test_indices
    ):
        global_index = int(
            global_index
        )

        predicted_label = int(
            test_labels[
                test_position
            ]
        )

        prediction_strength = float(
            test_strengths[
                test_position
            ]
        )

        score_row = centroid_scores[
            test_position
        ]

        ranked_cluster_ids = rank_clusters(
            score_row=score_row,
            cluster_ids=cluster_ids,
        )

        nearest_centroid_cluster = (
            ranked_cluster_ids[0]
        )

        if predicted_label >= 0:
            primary_cluster = (
                predicted_label
            )

            assignment_method = (
                "HDBSCAN approximate_predict"
            )

            direct_count += 1
        else:
            primary_cluster = (
                nearest_centroid_cluster
            )

            assignment_method = (
                "noise → en yakın centroid"
            )

            fallback_count += 1

        secondary_cluster = (
            select_secondary_cluster(
                primary_cluster=primary_cluster,
                ranked_cluster_ids=(
                    ranked_cluster_ids
                ),
            )
        )

        primary_similarity = float(
            score_row[
                cluster_position[
                    primary_cluster
                ]
            ]
        )

        secondary_similarity = float(
            score_row[
                cluster_position[
                    secondary_cluster
                ]
            ]
        )

        similarity_margin = (
            primary_similarity
            - secondary_similarity
        )

        primary_subject_key = (
            cluster_info[
                primary_cluster
            ]["dominant_subject_key"]
        )

        secondary_subject_key = (
            cluster_info[
                secondary_cluster
            ]["dominant_subject_key"]
        )

        known_subjects = (
            article_subject_sets[
                global_index
            ]
        )

        if known_subjects:
            labeled_test_count += 1

            top1_match = (
                bool(
                    primary_subject_key
                )
                and primary_subject_key
                in known_subjects
            )

            top2_match = (
                top1_match
                or (
                    bool(
                        secondary_subject_key
                    )
                    and secondary_subject_key
                    in known_subjects
                )
            )

            if top1_match:
                total_top1_count += 1

            if top2_match:
                total_top2_count += 1

            if predicted_label >= 0:
                direct_labeled_count += 1

                if top1_match:
                    direct_top1_count += 1

                if top2_match:
                    direct_top2_count += 1
            else:
                fallback_labeled_count += 1

                if top1_match:
                    fallback_top1_count += 1

                if top2_match:
                    fallback_top2_count += 1
        else:
            top1_match = None
            top2_match = None

        result_rows.append(
            {
                "test_position": (
                    test_position
                ),
                "global_row_index": (
                    global_index
                ),
                "article_id": articles[
                    global_index
                ].get(
                    "article_id",
                    "",
                ),
                "publication_year": articles[
                    global_index
                ].get(
                    "publication_year",
                    "",
                ),
                "title_tr": articles[
                    global_index
                ].get(
                    "title_tr",
                    "",
                ),
                "hdbscan_status": (
                    "clusterlandı"
                    if predicted_label >= 0
                    else "noise / belirsiz"
                ),
                "hdbscan_probability": (
                    prediction_strength
                ),
                "assignment_method": (
                    assignment_method
                ),
                "primary_cluster": (
                    primary_cluster
                ),
                "primary_topic": (
                    cluster_info[
                        primary_cluster
                    ]["dominant_subject_name"]
                ),
                "primary_centroid_similarity": (
                    primary_similarity
                ),
                "secondary_cluster": (
                    secondary_cluster
                ),
                "secondary_topic": (
                    cluster_info[
                        secondary_cluster
                    ]["dominant_subject_name"]
                ),
                "secondary_centroid_similarity": (
                    secondary_similarity
                ),
                "similarity_margin": (
                    similarity_margin
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

    summary = {
        "random_seed": RANDOM_SEED,
        "training_article_count": (
            ARTICLE_COUNT
            - len(test_indices)
        ),
        "test_article_count": len(
            test_indices
        ),
        "training_cluster_count": len(
            cluster_ids
        ),
        "test_direct_hdbscan_count": (
            direct_count
        ),
        "test_direct_hdbscan_rate": (
            direct_count
            / len(test_indices)
        ),
        "test_centroid_fallback_count": (
            fallback_count
        ),
        "test_centroid_fallback_rate": (
            fallback_count
            / len(test_indices)
        ),
        "test_subject_labeled_count": (
            labeled_test_count
        ),
        "holdout_top1_metadata_consistency": (
            total_top1_count
            / labeled_test_count
            if labeled_test_count
            else 0.0
        ),
        "holdout_top2_metadata_consistency": (
            total_top2_count
            / labeled_test_count
            if labeled_test_count
            else 0.0
        ),
        "direct_top1_metadata_consistency": (
            direct_top1_count
            / direct_labeled_count
            if direct_labeled_count
            else 0.0
        ),
        "direct_top2_metadata_consistency": (
            direct_top2_count
            / direct_labeled_count
            if direct_labeled_count
            else 0.0
        ),
        "fallback_top1_metadata_consistency": (
            fallback_top1_count
            / fallback_labeled_count
            if fallback_labeled_count
            else 0.0
        ),
        "fallback_top2_metadata_consistency": (
            fallback_top2_count
            / fallback_labeled_count
            if fallback_labeled_count
            else 0.0
        ),
        "important_note": (
            "Cluster konu adları yalnızca 800 eğitim "
            "makalesinin subject metadata alanlarından "
            "oluşturulmuştur. Test makalelerinin subjectleri "
            "sadece sonradan değerlendirme amacıyla kullanılmıştır."
        ),
    }

    return (
        result_rows,
        summary,
    )


def save_cluster_dictionary(
    cluster_ids: List[int],
    cluster_info: Dict[int, Dict[str, Any]],
) -> Path:
    """Eğitim verisinden oluşturulan cluster sözlüğünü kaydeder."""

    output_path = (
        get_output_directory()
        / "day27_holdout_cluster_dictionary.csv"
    )

    rows = [
        cluster_info[
            cluster_id
        ]
        for cluster_id in cluster_ids
    ]

    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    return output_path


def save_predictions(
    result_rows: List[Dict[str, Any]],
) -> Path:
    """200 test makalesinin tahminlerini kaydeder."""

    output_path = (
        get_output_directory()
        / "day27_holdout_predictions.csv"
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                result_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            result_rows
        )

    return output_path


def save_summary(
    summary: Dict[str, Any],
) -> Path:
    """Holdout özetini JSON olarak kaydeder."""

    output_path = (
        get_output_directory()
        / "day27_holdout_summary.json"
    )

    output_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_path


def print_summary(
    summary: Dict[str, Any],
) -> None:
    """Holdout sonucunu terminalde gösterir."""

    print("\n" + "=" * 90)
    print("DAY 27 — HOLDOUT TEST SONUCU")
    print("=" * 90)

    print(
        f"\nEğitim makalesi          : "
        f"{summary['training_article_count']}"
    )

    print(
        f"Test makalesi            : "
        f"{summary['test_article_count']}"
    )

    print(
        f"Eğitimde bulunan cluster : "
        f"{summary['training_cluster_count']}"
    )

    print(
        f"\nTest doğrudan HDBSCAN    : "
        f"{summary['test_direct_hdbscan_count']} "
        f"(%{summary['test_direct_hdbscan_rate'] * 100:.2f})"
    )

    print(
        f"Test centroid fallback   : "
        f"{summary['test_centroid_fallback_count']} "
        f"(%{summary['test_centroid_fallback_rate'] * 100:.2f})"
    )

    print(
        f"\nHoldout Top-1            : "
        f"%{summary['holdout_top1_metadata_consistency'] * 100:.2f}"
    )

    print(
        f"Holdout Top-2            : "
        f"%{summary['holdout_top2_metadata_consistency'] * 100:.2f}"
    )

    print(
        f"\nDoğrudan HDBSCAN Top-1   : "
        f"%{summary['direct_top1_metadata_consistency'] * 100:.2f}"
    )

    print(
        f"Fallback Top-1           : "
        f"%{summary['fallback_top1_metadata_consistency'] * 100:.2f}"
    )

    print(
        "\nBu kez test subjectleri cluster isimlerini "
        "oluşturmak için kullanılmadı."
    )


def main() -> None:
    print("=" * 90)
    print("DAY 27 — 800/200 HOLDOUT DEĞERLENDİRMESİ")
    print("=" * 90)

    articles = load_articles()
    embeddings = load_embeddings()

    (
        article_subject_sets,
        subject_display_names,
    ) = build_subject_information(
        articles
    )

    (
        train_indices,
        test_indices,
    ) = create_train_test_split(
        articles
    )

    train_embeddings = embeddings[
        train_indices
    ]

    test_embeddings = embeddings[
        test_indices
    ]

    (
        _,
        _,
        train_labels,
        test_labels,
        test_strengths,
    ) = train_umap_and_hdbscan(
        train_embeddings=(
            train_embeddings
        ),
        test_embeddings=(
            test_embeddings
        ),
    )

    (
        cluster_ids,
        cluster_info,
        centroid_matrix,
    ) = build_training_cluster_dictionary(
        train_indices=train_indices,
        train_embeddings=(
            train_embeddings
        ),
        train_labels=train_labels,
        article_subject_sets=(
            article_subject_sets
        ),
        subject_display_names=(
            subject_display_names
        ),
    )

    (
        result_rows,
        summary,
    ) = evaluate_holdout(
        articles=articles,
        test_indices=test_indices,
        test_embeddings=(
            test_embeddings
        ),
        test_labels=test_labels,
        test_strengths=(
            test_strengths
        ),
        cluster_ids=cluster_ids,
        cluster_info=cluster_info,
        centroid_matrix=(
            centroid_matrix
        ),
        article_subject_sets=(
            article_subject_sets
        ),
        subject_display_names=(
            subject_display_names
        ),
    )

    cluster_path = save_cluster_dictionary(
        cluster_ids=cluster_ids,
        cluster_info=cluster_info,
    )

    predictions_path = (
        save_predictions(
            result_rows
        )
    )

    summary_path = save_summary(
        summary
    )

    print_summary(
        summary
    )

    print("\n" + "=" * 90)
    print("DOSYALAR")
    print("=" * 90)

    print(
        f"\nEğitim cluster sözlüğü:\n"
        f"{cluster_path}"
    )

    print(
        f"\nHoldout tahminleri:\n"
        f"{predictions_path}"
    )

    print(
        f"\nHoldout teknik özeti:\n"
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()