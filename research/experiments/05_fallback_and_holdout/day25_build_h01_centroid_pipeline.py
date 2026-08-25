import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Day 24'te yazdığımız veri okuma ve cluster hazırlama
# fonksiyonlarını tekrar kullanıyoruz.
from day24_compare_noise_assignment_methods import (
    build_cluster_representations,
    build_subject_information,
    get_output_directory,
    load_articles,
    load_embeddings,
    load_h01_assignments,
    validate_alignment,
)


ARTICLE_COUNT = 1000
PIPELINE_NAME = "HDBSCAN H01 + centroid noise assignment"


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


def rank_clusters(
    score_row: np.ndarray,
    cluster_ids: List[int],
) -> List[int]:
    """
    Centroid benzerlik skorlarını büyükten küçüğe
    sıralayıp cluster ID listesi döndürür.
    """

    ranked_positions = np.argsort(
        score_row
    )[::-1]

    return [
        cluster_ids[int(position)]
        for position in ranked_positions
    ]


def get_secondary_cluster(
    primary_cluster_id: int,
    ranked_cluster_ids: List[int],
) -> int:
    """Birincil clusterdan farklı en yakın clusterı seçer."""

    for cluster_id in ranked_cluster_ids:
        if cluster_id != primary_cluster_id:
            return cluster_id

    raise RuntimeError(
        "İkinci cluster bulunamadı."
    )


def safe_percentile(
    values: np.ndarray,
    percentile: float,
) -> float:
    """Boş olmayan sayı dizisi için yüzdelik hesaplar."""

    if values.size == 0:
        return 0.0

    return float(
        np.percentile(
            values,
            percentile,
        )
    )


def classify_relative_confidence(
    original_hdbscan_label: int,
    hdbscan_probability: float,
    centroid_agrees_with_hdbscan: Any,
    similarity_margin: float,
    direct_probability_q25: float,
    direct_probability_q75: float,
    direct_margin_q25: float,
    direct_margin_q75: float,
    noise_margin_q25: float,
    noise_margin_q75: float,
) -> str:
    """
    Pilot veri içindeki göreli güven seviyesini üretir.

    Bu değer kalibre edilmiş bir olasılık değildir.
    """

    # Doğrudan HDBSCAN tarafından clusterlanan makale.
    if original_hdbscan_label >= 0:
        if (
            centroid_agrees_with_hdbscan is False
            or hdbscan_probability <= direct_probability_q25
            or similarity_margin <= direct_margin_q25
        ):
            return "düşük"

        if (
            centroid_agrees_with_hdbscan is True
            and hdbscan_probability >= direct_probability_q75
            and similarity_margin >= direct_margin_q75
        ):
            return "yüksek"

        return "orta"

    # Başlangıçta noise olan makale.
    if similarity_margin <= noise_margin_q25:
        return "düşük"

    if similarity_margin >= noise_margin_q75:
        return "yüksek"

    return "orta"


def classify_topic_structure(
    original_hdbscan_label: int,
    centroid_agrees_with_hdbscan: Any,
    similarity_margin: float,
    direct_margin_q25: float,
    direct_margin_q75: float,
    noise_margin_q25: float,
    noise_margin_q75: float,
) -> str:
    """Makalenin konu uzayındaki göreli durumunu yorumlar."""

    if original_hdbscan_label >= 0:
        if (
            centroid_agrees_with_hdbscan is False
            or similarity_margin <= direct_margin_q25
        ):
            return "çok alanlı / cluster sınırı adayı"

        if similarity_margin >= direct_margin_q75:
            return "belirgin HDBSCAN konu çekirdeği"

        return "orta düzeyde ayrışmış HDBSCAN konusu"

    if similarity_margin <= noise_margin_q25:
        return "noise ve çok alanlı / geçiş bölgesi adayı"

    if similarity_margin >= noise_margin_q75:
        return "noise iken belirgin centroid konusuna bağlandı"

    return "noise iken en yakın centroid konusuna bağlandı"


def build_raw_assignment_rows(
    articles: List[Dict[str, Any]],
    embeddings: np.ndarray,
    assignments: List[Dict[str, Any]],
    cluster_ids: List[int],
    cluster_info: Dict[int, Dict[str, Any]],
    centroid_matrix: np.ndarray,
) -> Tuple[
    List[Dict[str, Any]],
    np.ndarray,
]:
    """
    Bütün makaleler için birincil ve ikincil centroid
    benzerliklerini hesaplar.

    Doğrudan HDBSCAN clusterlanan makalelerin birincil
    clusterı değiştirilmez.

    Noise makalelerin birincil clusterı en yakın centroid
    üzerinden seçilir.
    """

    score_matrix = (
        embeddings
        @ centroid_matrix.T
    )

    cluster_position = {
        cluster_id: position
        for position, cluster_id
        in enumerate(cluster_ids)
    }

    raw_rows: List[
        Dict[str, Any]
    ] = []

    for row_index in range(
        ARTICLE_COUNT
    ):
        article = articles[
            row_index
        ]

        assignment = assignments[
            row_index
        ]

        original_label = int(
            assignment["label"]
        )

        score_row = score_matrix[
            row_index
        ]

        ranked_cluster_ids = (
            rank_clusters(
                score_row=score_row,
                cluster_ids=cluster_ids,
            )
        )

        nearest_centroid_cluster = (
            ranked_cluster_ids[0]
        )

        if original_label >= 0:
            # HDBSCAN'in doğal cluster atamasını koruyoruz.
            primary_cluster_id = (
                original_label
            )

            assignment_method = (
                "doğrudan HDBSCAN H01"
            )

            centroid_agrees = (
                nearest_centroid_cluster
                == original_label
            )
        else:
            # Noise makaleyi en yakın centroid kümesine bağlıyoruz.
            primary_cluster_id = (
                nearest_centroid_cluster
            )

            assignment_method = (
                "noise → en yakın normalize centroid"
            )

            centroid_agrees = None

        secondary_cluster_id = (
            get_secondary_cluster(
                primary_cluster_id=(
                    primary_cluster_id
                ),
                ranked_cluster_ids=(
                    ranked_cluster_ids
                ),
            )
        )

        primary_score = float(
            score_row[
                cluster_position[
                    primary_cluster_id
                ]
            ]
        )

        secondary_score = float(
            score_row[
                cluster_position[
                    secondary_cluster_id
                ]
            ]
        )

        similarity_margin = (
            primary_score
            - secondary_score
        )

        primary_info = cluster_info[
            primary_cluster_id
        ]

        secondary_info = cluster_info[
            secondary_cluster_id
        ]

        raw_rows.append(
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
                "title_tr": article.get(
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
                "hdbscan_probability": float(
                    assignment["probability"]
                ),
                "hdbscan_outlier_score": float(
                    assignment["outlier_score"]
                ),
                "assignment_method": (
                    assignment_method
                ),
                "nearest_centroid_cluster": (
                    nearest_centroid_cluster
                ),
                "centroid_agrees_with_hdbscan": (
                    centroid_agrees
                ),
                "primary_cluster": (
                    primary_cluster_id
                ),
                "primary_topic": (
                    primary_info[
                        "dominant_subject_name"
                    ]
                ),
                "primary_subject_key": (
                    primary_info[
                        "dominant_subject_key"
                    ]
                ),
                "primary_centroid_similarity": (
                    primary_score
                ),
                "primary_medoid_article_id": (
                    primary_info[
                        "medoid_article_id"
                    ]
                ),
                "primary_medoid_title": (
                    primary_info[
                        "medoid_title"
                    ]
                ),
                "secondary_cluster": (
                    secondary_cluster_id
                ),
                "secondary_topic": (
                    secondary_info[
                        "dominant_subject_name"
                    ]
                ),
                "secondary_subject_key": (
                    secondary_info[
                        "dominant_subject_key"
                    ]
                ),
                "secondary_centroid_similarity": (
                    secondary_score
                ),
                "similarity_margin": (
                    similarity_margin
                ),
            }
        )

    return (
        raw_rows,
        score_matrix,
    )


def calculate_thresholds(
    raw_rows: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Doğrudan clusterlanan ve noise makaleler için
    ayrı göreli eşikler hesaplar.
    """

    direct_margins = np.array(
        [
            float(row["similarity_margin"])
            for row in raw_rows
            if row[
                "original_hdbscan_status"
            ] == "clusterlandı"
        ],
        dtype=np.float32,
    )

    noise_margins = np.array(
        [
            float(row["similarity_margin"])
            for row in raw_rows
            if row[
                "original_hdbscan_status"
            ] == "noise / belirsiz"
        ],
        dtype=np.float32,
    )

    direct_probabilities = np.array(
        [
            float(row["hdbscan_probability"])
            for row in raw_rows
            if row[
                "original_hdbscan_status"
            ] == "clusterlandı"
        ],
        dtype=np.float32,
    )

    return {
        "direct_margin_q25": (
            safe_percentile(
                direct_margins,
                25,
            )
        ),
        "direct_margin_median": (
            safe_percentile(
                direct_margins,
                50,
            )
        ),
        "direct_margin_q75": (
            safe_percentile(
                direct_margins,
                75,
            )
        ),
        "noise_margin_q25": (
            safe_percentile(
                noise_margins,
                25,
            )
        ),
        "noise_margin_median": (
            safe_percentile(
                noise_margins,
                50,
            )
        ),
        "noise_margin_q75": (
            safe_percentile(
                noise_margins,
                75,
            )
        ),
        "direct_probability_q25": (
            safe_percentile(
                direct_probabilities,
                25,
            )
        ),
        "direct_probability_median": (
            safe_percentile(
                direct_probabilities,
                50,
            )
        ),
        "direct_probability_q75": (
            safe_percentile(
                direct_probabilities,
                75,
            )
        ),
    }


def calculate_metadata_consistency(
    primary_subject_key: str,
    secondary_subject_key: str,
    known_subjects: Set[str],
) -> Tuple[Any, Any]:
    """Top-1 ve Top-2 metadata tutarlılığını hesaplar."""

    if not known_subjects:
        return None, None

    top1_matches = (
        bool(primary_subject_key)
        and primary_subject_key
        in known_subjects
    )

    top2_matches = (
        top1_matches
        or (
            bool(secondary_subject_key)
            and secondary_subject_key
            in known_subjects
        )
    )

    return (
        top1_matches,
        top2_matches,
    )


def build_final_rows(
    raw_rows: List[Dict[str, Any]],
    article_subject_sets: List[Set[str]],
    subject_display_names: Dict[str, str],
    thresholds: Dict[str, float],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    """Göreli güven ve metadata tutarlılığını ekler."""

    final_rows: List[
        Dict[str, Any]
    ] = []

    confidence_counter = Counter()
    structure_counter = Counter()

    total_labeled_count = 0
    total_top1_match = 0
    total_top2_match = 0

    direct_labeled_count = 0
    direct_top1_match = 0
    direct_top2_match = 0

    noise_labeled_count = 0
    noise_top1_match = 0
    noise_top2_match = 0

    direct_count = 0
    noise_count = 0
    centroid_recovery_count = 0

    for raw_row in raw_rows:
        row_index = int(
            raw_row["row_index"]
        )

        original_label = int(
            raw_row[
                "original_hdbscan_cluster"
            ]
        )

        is_direct = (
            original_label >= 0
        )

        if is_direct:
            direct_count += 1

            if (
                raw_row[
                    "centroid_agrees_with_hdbscan"
                ]
                is True
            ):
                centroid_recovery_count += 1
        else:
            noise_count += 1

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
                centroid_agrees_with_hdbscan=(
                    raw_row[
                        "centroid_agrees_with_hdbscan"
                    ]
                ),
                similarity_margin=float(
                    raw_row[
                        "similarity_margin"
                    ]
                ),
                direct_probability_q25=(
                    thresholds[
                        "direct_probability_q25"
                    ]
                ),
                direct_probability_q75=(
                    thresholds[
                        "direct_probability_q75"
                    ]
                ),
                direct_margin_q25=(
                    thresholds[
                        "direct_margin_q25"
                    ]
                ),
                direct_margin_q75=(
                    thresholds[
                        "direct_margin_q75"
                    ]
                ),
                noise_margin_q25=(
                    thresholds[
                        "noise_margin_q25"
                    ]
                ),
                noise_margin_q75=(
                    thresholds[
                        "noise_margin_q75"
                    ]
                ),
            )
        )

        topic_structure = (
            classify_topic_structure(
                original_hdbscan_label=(
                    original_label
                ),
                centroid_agrees_with_hdbscan=(
                    raw_row[
                        "centroid_agrees_with_hdbscan"
                    ]
                ),
                similarity_margin=float(
                    raw_row[
                        "similarity_margin"
                    ]
                ),
                direct_margin_q25=(
                    thresholds[
                        "direct_margin_q25"
                    ]
                ),
                direct_margin_q75=(
                    thresholds[
                        "direct_margin_q75"
                    ]
                ),
                noise_margin_q25=(
                    thresholds[
                        "noise_margin_q25"
                    ]
                ),
                noise_margin_q75=(
                    thresholds[
                        "noise_margin_q75"
                    ]
                ),
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

        (
            top1_matches,
            top2_matches,
        ) = calculate_metadata_consistency(
            primary_subject_key=str(
                raw_row[
                    "primary_subject_key"
                ]
            ),
            secondary_subject_key=str(
                raw_row[
                    "secondary_subject_key"
                ]
            ),
            known_subjects=known_subjects,
        )

        if known_subjects:
            total_labeled_count += 1

            if top1_matches:
                total_top1_match += 1

            if top2_matches:
                total_top2_match += 1

            if is_direct:
                direct_labeled_count += 1

                if top1_matches:
                    direct_top1_match += 1

                if top2_matches:
                    direct_top2_match += 1
            else:
                noise_labeled_count += 1

                if top1_matches:
                    noise_top1_match += 1

                if top2_matches:
                    noise_top2_match += 1

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
                    top1_matches
                ),
                "top2_topics_match_metadata": (
                    top2_matches
                ),
            }
        )

    summary = {
        "pipeline_name": (
            PIPELINE_NAME
        ),
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
        "noise_assignment_method": (
            "normalize H01 cluster centroidi"
        ),
        "cluster_count": 33,
        "directly_clustered_count": (
            direct_count
        ),
        "directly_clustered_rate": (
            direct_count
            / ARTICLE_COUNT
        ),
        "noise_centroid_assigned_count": (
            noise_count
        ),
        "noise_centroid_assigned_rate": (
            noise_count
            / ARTICLE_COUNT
        ),
        "final_output_coverage": 1.0,
        "centroid_direct_cluster_recovery_rate": (
            centroid_recovery_count
            / direct_count
            if direct_count
            else 0.0
        ),
        "subject_labeled_article_count": (
            total_labeled_count
        ),
        "metadata_top1_consistency_rate": (
            total_top1_match
            / total_labeled_count
            if total_labeled_count
            else 0.0
        ),
        "metadata_top2_consistency_rate": (
            total_top2_match
            / total_labeled_count
            if total_labeled_count
            else 0.0
        ),
        "direct_metadata_top1_consistency_rate": (
            direct_top1_match
            / direct_labeled_count
            if direct_labeled_count
            else 0.0
        ),
        "direct_metadata_top2_consistency_rate": (
            direct_top2_match
            / direct_labeled_count
            if direct_labeled_count
            else 0.0
        ),
        "noise_metadata_top1_consistency_rate": (
            noise_top1_match
            / noise_labeled_count
            if noise_labeled_count
            else 0.0
        ),
        "noise_metadata_top2_consistency_rate": (
            noise_top2_match
            / noise_labeled_count
            if noise_labeled_count
            else 0.0
        ),
        "confidence_distribution": dict(
            confidence_counter
        ),
        "topic_structure_distribution": dict(
            structure_counter
        ),
        "relative_thresholds": (
            thresholds
        ),
        "important_note": (
            "Cluster konu adları baskın TR Dizin subject "
            "metadata alanlarından türetilen geçici adlardır. "
            "Metadata tutarlılığı bağımsız test başarımı "
            "değildir. Güven seviyeleri kalibre edilmiş "
            "olasılıklar değildir."
        ),
    }

    return (
        final_rows,
        summary,
    )


def save_cluster_dictionary(
    cluster_ids: List[int],
    cluster_info: Dict[int, Dict[str, Any]],
) -> Path:
    """33 H01 clusterının konu sözlüğünü kaydeder."""

    output_path = (
        get_output_directory()
        / "day25_h01_centroid_cluster_dictionary.csv"
    )

    fieldnames = [
        "cluster_id",
        "cluster_size",
        "provisional_topic",
        "dominant_subject_key",
        "dominant_subject_count",
        "subject_labeled_count",
        "subject_purity",
        "medoid_article_id",
        "medoid_title",
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
                    "provisional_topic": (
                        info[
                            "dominant_subject_name"
                        ]
                    ),
                    "dominant_subject_key": (
                        info[
                            "dominant_subject_key"
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
                }
            )

    return output_path


def save_final_assignments(
    final_rows: List[Dict[str, Any]],
) -> Path:
    """Bütün makalelerin final konu çıktılarını kaydeder."""

    output_path = (
        get_output_directory()
        / "day25_h01_centroid_final_topic_assignments.csv"
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


def save_noise_assignments(
    final_rows: List[Dict[str, Any]],
) -> Path:
    """Centroidle atanan 230 noise makaleyi ayrı kaydeder."""

    output_path = (
        get_output_directory()
        / "day25_h01_centroid_noise_assignments.csv"
    )

    noise_rows = [
        row
        for row in final_rows
        if row[
            "original_hdbscan_status"
        ] == "noise / belirsiz"
    ]

    # En belirsiz makaleler üstte yer alsın.
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
        "primary_centroid_similarity",
        "secondary_cluster",
        "secondary_topic",
        "secondary_centroid_similarity",
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
        / "day25_h01_centroid_pipeline_summary.json"
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
    """Göreli güven dağılımı grafiğini oluşturur."""

    output_path = (
        get_output_directory()
        / "day25_h01_centroid_confidence_distribution.png"
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
            confidence,
            0,
        )
        for confidence
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
        "H01 + Centroid Pipeline — Göreli Güven Dağılımı"
    )

    plt.xlabel(
        "Göreli güven"
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
    """Okunabilir final pilot raporunu oluşturur."""

    output_path = (
        get_output_directory()
        / "day25_h01_centroid_pipeline_report.md"
    )

    confidence = summary[
        "confidence_distribution"
    ]

    topic_structures = summary[
        "topic_structure_distribution"
    ]

    lines: List[str] = [
        "# HDBSCAN H01 + Centroid Final Pilot Pipeline",
        "",
        "## Yöntem",
        "",
        "- Girdi: Türkçe abstract",
        "- Embedding: TR-MTEB",
        "- Ana clustering: HDBSCAN H01",
        "- Doğrudan clusterlanan makaleler: H01 kümelerinde tutuldu",
        (
            "- Noise makaleler: normalize cluster centroidleriyle "
            "birincil ve ikincil konuya bağlandı"
        ),
        "- Medoidler yalnızca kümeyi açıklayan gerçek temsilci makale olarak tutuldu",
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
            f"- Noise iken centroidle atanan: "
            f"{summary['noise_centroid_assigned_count']} "
            f"(%{summary['noise_centroid_assigned_rate'] * 100:.2f})"
        ),
        "- Final çıktı kapsamı: %100",
        "",
        "## Centroid kontrolü",
        "",
        (
            f"- H01 doğrudan cluster geri-bulma oranı: "
            f"%{summary['centroid_direct_cluster_recovery_rate'] * 100:.2f}"
        ),
        "",
        "## Metadata ile keşifsel tutarlılık",
        "",
        (
            f"- Genel Top-1: "
            f"%{summary['metadata_top1_consistency_rate'] * 100:.2f}"
        ),
        (
            f"- Genel Top-2: "
            f"%{summary['metadata_top2_consistency_rate'] * 100:.2f}"
        ),
        (
            f"- Doğrudan clusterlanan Top-1: "
            f"%{summary['direct_metadata_top1_consistency_rate'] * 100:.2f}"
        ),
        (
            f"- Doğrudan clusterlanan Top-2: "
            f"%{summary['direct_metadata_top2_consistency_rate'] * 100:.2f}"
        ),
        (
            f"- Noise centroid Top-1: "
            f"%{summary['noise_metadata_top1_consistency_rate'] * 100:.2f}"
        ),
        (
            f"- Noise centroid Top-2: "
            f"%{summary['noise_metadata_top2_consistency_rate'] * 100:.2f}"
        ),
        "",
        (
            "Bu değerler bağımsız test başarımı değildir. "
            "Konu adları da subject metadata üzerinden "
            "türetildiği için keşifsel tutarlılık göstergesidir."
        ),
        "",
        "## Göreli güven dağılımı",
        "",
        (
            f"- Yüksek: "
            f"{confidence.get('yüksek', 0)}"
        ),
        (
            f"- Orta: "
            f"{confidence.get('orta', 0)}"
        ),
        (
            f"- Düşük: "
            f"{confidence.get('düşük', 0)}"
        ),
        "",
        "## Konu yapısı dağılımı",
        "",
    ]

    for structure_name, count in (
        topic_structures.items()
    ):
        lines.append(
            f"- {structure_name}: {count}"
        )

    lines.extend(
        [
            "",
            "## Sınırlamalar",
            "",
            "- Cluster konu adları otomatik ve geçicidir.",
            "- Subject metadata ground truth olarak kabul edilmemelidir.",
            "- Noise centroid ataması yeni bir HDBSCAN ataması değildir.",
            "- Göreli güven değeri kalibre edilmiş olasılık değildir.",
            "- 50.000 makalelik ölçekte parametreler yeniden doğrulanmalıdır.",
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


def print_summary(
    summary: Dict[str, Any],
) -> None:
    """Terminalde final pilot özetini gösterir."""

    print("\n" + "=" * 90)
    print("FINAL H01 + CENTROID PIPELINE ÖZETİ")
    print("=" * 90)

    print(
        f"\nToplam makale                : "
        f"{summary['article_count']}"
    )

    print(
        f"H01 doğrudan clusterlanan    : "
        f"{summary['directly_clustered_count']} "
        f"(%{summary['directly_clustered_rate'] * 100:.2f})"
    )

    print(
        f"Noise iken centroidle atanan : "
        f"{summary['noise_centroid_assigned_count']} "
        f"(%{summary['noise_centroid_assigned_rate'] * 100:.2f})"
    )

    print(
        "\nFinal çıktı üretilen         : "
        "1000 (%100)"
    )

    print(
        f"\nCentroid H01 geri-bulma      : "
        f"%{summary['centroid_direct_cluster_recovery_rate'] * 100:.2f}"
    )

    print(
        f"\nGenel metadata Top-1         : "
        f"%{summary['metadata_top1_consistency_rate'] * 100:.2f}"
    )

    print(
        f"Genel metadata Top-2         : "
        f"%{summary['metadata_top2_consistency_rate'] * 100:.2f}"
    )

    print(
        f"\nDoğrudan H01 Top-1           : "
        f"%{summary['direct_metadata_top1_consistency_rate'] * 100:.2f}"
    )

    print(
        f"Noise centroid Top-1         : "
        f"%{summary['noise_metadata_top1_consistency_rate'] * 100:.2f}"
    )

    print(
        f"Noise centroid Top-2         : "
        f"%{summary['noise_metadata_top2_consistency_rate'] * 100:.2f}"
    )

    print(
        "\nNot: Metadata oranları bağımsız "
        "test başarımı değildir."
    )


def main() -> None:
    print("=" * 90)
    print("DAY 25 — HDBSCAN H01 + CENTROID FINAL PİLOT")
    print("=" * 90)

    articles = load_articles()
    embeddings = load_embeddings()
    assignments = load_h01_assignments()

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

    centroid_matrix = representations[
        "centroid"
    ]

    (
        raw_rows,
        _,
    ) = build_raw_assignment_rows(
        articles=articles,
        embeddings=embeddings,
        assignments=assignments,
        cluster_ids=cluster_ids,
        cluster_info=cluster_info,
        centroid_matrix=centroid_matrix,
    )

    thresholds = calculate_thresholds(
        raw_rows
    )

    (
        final_rows,
        summary,
    ) = build_final_rows(
        raw_rows=raw_rows,
        article_subject_sets=(
            article_subject_sets
        ),
        subject_display_names=(
            subject_display_names
        ),
        thresholds=thresholds,
    )

    dictionary_path = (
        save_cluster_dictionary(
            cluster_ids=cluster_ids,
            cluster_info=cluster_info,
        )
    )

    assignments_path = (
        save_final_assignments(
            final_rows
        )
    )

    noise_path = (
        save_noise_assignments(
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

    print_summary(
        summary
    )

    print("\n" + "=" * 90)
    print("DOSYALAR")
    print("=" * 90)

    print(
        f"\nCluster konu sözlüğü:\n"
        f"{dictionary_path}"
    )

    print(
        f"\nMakale bazlı final çıktılar:\n"
        f"{assignments_path}"
    )

    print(
        f"\nCentroidle atanan noise makaleler:\n"
        f"{noise_path}"
    )

    print(
        f"\nTeknik özet:\n"
        f"{summary_path}"
    )

    print(
        f"\nGöreli güven grafiği:\n"
        f"{chart_path}"
    )

    print(
        f"\nOkunabilir rapor:\n"
        f"{report_path}"
    )


if __name__ == "__main__":
    main()