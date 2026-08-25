import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CANDIDATE_CONFIG_IDS = [
    "H01",
    "H16",
    "H18",
]


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


def load_umap_coordinates() -> List[Dict[str, Any]]:
    """
    Day 16'da oluşturulan 2D UMAP koordinatlarını okur.

    Bu koordinatlar yalnızca görselleştirme içindir.
    HDBSCAN bu koordinatlar üzerinde çalıştırılmamıştır.
    """

    input_path = (
        get_project_root()
        / "research" / "outputs"
        / "day16_tr_mteb_umap_coordinates.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"UMAP koordinat dosyası bulunamadı:\n{input_path}"
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
            "umap_x",
            "umap_y",
            "cluster_id",
            "silhouette",
            "title_tr",
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
                "UMAP CSV dosyasında eksik sütunlar var: "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            rows.append(
                {
                    "row_index": int(row["row_index"]),
                    "article_id": row["article_id"],
                    "umap_x": float(row["umap_x"]),
                    "umap_y": float(row["umap_y"]),
                    "kmeans_cluster_id": int(
                        row["cluster_id"]
                    ),
                    "kmeans_silhouette": float(
                        row["silhouette"]
                    ),
                    "title_tr": row["title_tr"],
                }
            )

    rows.sort(
        key=lambda row: row["row_index"]
    )

    if len(rows) != 1000:
        raise ValueError(
            "1.000 UMAP koordinatı bekleniyordu, "
            f"bulunan: {len(rows)}"
        )

    return rows


def load_hdbscan_assignments() -> List[Dict[str, Any]]:
    """Day 17'deki bütün HDBSCAN atamalarını okur."""

    input_path = (
        get_project_root()
        / "research" / "outputs"
        / "day17_hdbscan_all_assignments.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"HDBSCAN atama dosyası bulunamadı:\n{input_path}"
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
            "kmeans_cluster_id",
            "kmeans_silhouette",
        }

        for config_id in CANDIDATE_CONFIG_IDS:
            required_columns.update(
                {
                    f"{config_id}_label",
                    f"{config_id}_probability",
                    f"{config_id}_outlier_score",
                }
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
                "HDBSCAN CSV dosyasında eksik sütunlar var: "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            parsed_row: Dict[str, Any] = {
                "row_index": int(row["row_index"]),
                "article_id": row["article_id"],
                "kmeans_cluster_id": int(
                    row["kmeans_cluster_id"]
                ),
                "kmeans_silhouette": float(
                    row["kmeans_silhouette"]
                ),
            }

            for config_id in CANDIDATE_CONFIG_IDS:
                parsed_row[
                    f"{config_id}_label"
                ] = int(
                    row[f"{config_id}_label"]
                )

                parsed_row[
                    f"{config_id}_probability"
                ] = float(
                    row[f"{config_id}_probability"]
                )

                parsed_row[
                    f"{config_id}_outlier_score"
                ] = float(
                    row[f"{config_id}_outlier_score"]
                )

            rows.append(parsed_row)

    rows.sort(
        key=lambda row: row["row_index"]
    )

    if len(rows) != 1000:
        raise ValueError(
            "1.000 HDBSCAN ataması bekleniyordu, "
            f"bulunan: {len(rows)}"
        )

    return rows


def load_sweep_summaries() -> Dict[str, Dict[str, Any]]:
    """Day 17 parametre tarama özetini okur."""

    input_path = (
        get_project_root()
        / "research" / "outputs"
        / "day17_hdbscan_sweep_summary.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"HDBSCAN özet dosyası bulunamadı:\n{input_path}"
        )

    summaries: Dict[
        str,
        Dict[str, Any],
    ] = {}

    with input_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            config_id = row["config_id"]

            if config_id not in CANDIDATE_CONFIG_IDS:
                continue

            summaries[config_id] = {
                "config_id": config_id,
                "selection_method": row[
                    "selection_method"
                ],
                "min_cluster_size": int(
                    row["min_cluster_size"]
                ),
                "min_samples": int(
                    row["min_samples"]
                ),
                "cluster_count": int(
                    row["cluster_count"]
                ),
                "noise_count": int(
                    row["noise_count"]
                ),
                "noise_rate": float(
                    row["noise_rate"]
                ),
                "original_cosine_silhouette": float(
                    row["original_cosine_silhouette"]
                ),
                "mean_membership_probability": float(
                    row["mean_membership_probability"]
                ),
                "negative_capture_rate": float(
                    row["negative_capture_rate"]
                ),
            }

    missing_configs = (
        set(CANDIDATE_CONFIG_IDS)
        - set(summaries)
    )

    if missing_configs:
        raise ValueError(
            "Özet dosyasında aday yapılandırmalar bulunamadı: "
            + ", ".join(sorted(missing_configs))
        )

    return summaries


def validate_row_alignment(
    coordinate_rows: List[Dict[str, Any]],
    hdbscan_rows: List[Dict[str, Any]],
) -> None:
    """İki CSV dosyasındaki satırların aynı makalelere ait olduğunu doğrular."""

    for coordinate_row, hdbscan_row in zip(
        coordinate_rows,
        hdbscan_rows,
    ):
        if (
            coordinate_row["row_index"]
            != hdbscan_row["row_index"]
        ):
            raise ValueError(
                "Dosyalar arasında row_index uyuşmazlığı var."
            )

        if (
            coordinate_row["article_id"]
            != hdbscan_row["article_id"]
        ):
            raise ValueError(
                "Dosyalar arasında article_id uyuşmazlığı var."
            )


def calculate_cluster_label_positions(
    points_2d: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, Tuple[float, float]]:
    """
    Her HDBSCAN cluster numarası için
    etiketin yazılacağı medyan konumu hesaplar.
    """

    positions: Dict[
        int,
        Tuple[float, float],
    ] = {}

    valid_cluster_ids = sorted(
        {
            int(label)
            for label in labels
            if int(label) >= 0
        }
    )

    for cluster_id in valid_cluster_ids:
        cluster_points = points_2d[
            labels == cluster_id
        ]

        median_point = np.median(
            cluster_points,
            axis=0,
        )

        positions[cluster_id] = (
            float(median_point[0]),
            float(median_point[1]),
        )

    return positions


def create_candidate_plot(
    config_id: str,
    points_2d: np.ndarray,
    labels: np.ndarray,
    summary: Dict[str, Any],
) -> Path:
    """
    Bir HDBSCAN adayının clusterlarını
    Day 16 UMAP koordinatları üzerinde gösterir.
    """

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / f"day18_{config_id.lower()}_umap.png"
    )

    clustered_mask = labels >= 0
    noise_mask = labels == -1

    label_positions = (
        calculate_cluster_label_positions(
            points_2d=points_2d,
            labels=labels,
        )
    )

    plt.figure(
        figsize=(16, 11)
    )

    clustered_scatter = plt.scatter(
        points_2d[
            clustered_mask,
            0,
        ],
        points_2d[
            clustered_mask,
            1,
        ],
        c=labels[clustered_mask],
        s=30,
        alpha=0.78,
    )

    plt.scatter(
        points_2d[
            noise_mask,
            0,
        ],
        points_2d[
            noise_mask,
            1,
        ],
        marker="x",
        s=32,
        alpha=0.55,
        label=(
            f"Noise: "
            f"{int(np.sum(noise_mask))}"
        ),
    )

    colorbar = plt.colorbar(
        clustered_scatter
    )

    colorbar.set_label(
        "HDBSCAN Cluster ID"
    )

    for cluster_id, (
        x_position,
        y_position,
    ) in label_positions.items():
        plt.annotate(
            str(cluster_id),
            (
                x_position,
                y_position,
            ),
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "alpha": 0.75,
            },
        )

    plt.title(
        (
            f"{config_id} — HDBSCAN Clusterları\n"
            f"leaf | "
            f"min_cluster_size="
            f"{summary['min_cluster_size']} | "
            f"min_samples="
            f"{summary['min_samples']} | "
            f"cluster="
            f"{summary['cluster_count']} | "
            f"noise=%"
            f"{summary['noise_rate'] * 100:.1f} | "
            f"silhouette="
            f"{summary['original_cosine_silhouette']:.4f}"
        )
    )

    plt.xlabel(
        "Day 16 UMAP Boyutu 1"
    )

    plt.ylabel(
        "Day 16 UMAP Boyutu 2"
    )

    plt.legend()

    plt.figtext(
        0.5,
        0.015,
        (
            "Çarpı işaretleri HDBSCAN noise noktalarıdır. "
            "2D koordinatlar yalnızca görselleştirme içindir; "
            "clustering 10D UMAP verisinde yapılmıştır."
        ),
        ha="center",
    )

    plt.tight_layout(
        rect=(0, 0.035, 1, 1)
    )

    plt.savefig(
        output_path,
        dpi=180,
    )

    plt.close()

    return output_path


def calculate_candidate_comparison(
    config_id: str,
    labels: np.ndarray,
    probabilities: np.ndarray,
    kmeans_silhouettes: np.ndarray,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Aday yapılandırmanın noise ve üyelik özetini hesaplar."""

    noise_mask = labels == -1
    clustered_mask = labels >= 0

    kmeans_negative_mask = (
        kmeans_silhouettes < 0
    )

    kmeans_positive_mask = (
        kmeans_silhouettes >= 0
    )

    negative_noise_count = int(
        np.sum(
            noise_mask
            & kmeans_negative_mask
        )
    )

    positive_noise_count = int(
        np.sum(
            noise_mask
            & kmeans_positive_mask
        )
    )

    kmeans_negative_count = int(
        np.sum(kmeans_negative_mask)
    )

    kmeans_positive_count = int(
        np.sum(kmeans_positive_mask)
    )

    low_probability_count = int(
        np.sum(
            clustered_mask
            & (probabilities < 0.5)
        )
    )

    return {
        "config_id": config_id,
        "selection_method": summary[
            "selection_method"
        ],
        "min_cluster_size": summary[
            "min_cluster_size"
        ],
        "min_samples": summary[
            "min_samples"
        ],
        "cluster_count": summary[
            "cluster_count"
        ],
        "clustered_count": int(
            np.sum(clustered_mask)
        ),
        "noise_count": int(
            np.sum(noise_mask)
        ),
        "noise_rate": float(
            np.mean(noise_mask)
        ),
        "original_cosine_silhouette": summary[
            "original_cosine_silhouette"
        ],
        "mean_membership_probability": float(
            probabilities[
                clustered_mask
            ].mean()
        ),
        "median_membership_probability": float(
            np.median(
                probabilities[
                    clustered_mask
                ]
            )
        ),
        "kmeans_negative_count": (
            kmeans_negative_count
        ),
        "kmeans_negative_noise_count": (
            negative_noise_count
        ),
        "kmeans_negative_noise_rate": (
            negative_noise_count
            / kmeans_negative_count
            if kmeans_negative_count
            else 0.0
        ),
        "kmeans_positive_count": (
            kmeans_positive_count
        ),
        "kmeans_positive_noise_count": (
            positive_noise_count
        ),
        "kmeans_positive_noise_rate": (
            positive_noise_count
            / kmeans_positive_count
            if kmeans_positive_count
            else 0.0
        ),
        "clustered_probability_below_0_5": (
            low_probability_count
        ),
    }


def save_comparison_csv(
    comparison_rows: List[Dict[str, Any]],
) -> Path:
    """Üç adayın karşılaştırmasını CSV olarak kaydeder."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day18_hdbscan_candidate_comparison.csv"
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                comparison_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            comparison_rows
        )

    return output_path


def print_comparison(
    comparison_rows: List[Dict[str, Any]],
) -> None:
    """Aday karşılaştırmasını terminalde gösterir."""

    print("\n" + "=" * 105)
    print("HDBSCAN ADAY KARŞILAŞTIRMASI")
    print("=" * 105)

    header = (
        f"{'ID':5}"
        f"{'Cluster':>9}"
        f"{'Noise':>10}"
        f"{'Silhouette':>13}"
        f"{'Ort. üyelik':>14}"
        f"{'KMeans negatif→noise':>23}"
        f"{'KMeans pozitif→noise':>23}"
    )

    print("\n" + header)
    print("-" * len(header))

    for row in comparison_rows:
        print(
            f"{row['config_id']:5}"
            f"{row['cluster_count']:>9}"
            f"{row['noise_rate'] * 100:>9.1f}%"
            f"{row['original_cosine_silhouette']:>13.4f}"
            f"{row['mean_membership_probability']:>14.4f}"
            f"{row['kmeans_negative_noise_rate'] * 100:>22.1f}%"
            f"{row['kmeans_positive_noise_rate'] * 100:>22.1f}%"
        )


def main() -> None:
    print("=" * 80)
    print("DAY 18 — HDBSCAN ADAYLARINI GÖRSEL KARŞILAŞTIRMA")
    print("=" * 80)

    coordinate_rows = (
        load_umap_coordinates()
    )

    hdbscan_rows = (
        load_hdbscan_assignments()
    )

    summaries = (
        load_sweep_summaries()
    )

    validate_row_alignment(
        coordinate_rows=coordinate_rows,
        hdbscan_rows=hdbscan_rows,
    )

    points_2d = np.array(
        [
            [
                row["umap_x"],
                row["umap_y"],
            ]
            for row in coordinate_rows
        ],
        dtype=np.float32,
    )

    kmeans_silhouettes = np.array(
        [
            row["kmeans_silhouette"]
            for row in hdbscan_rows
        ],
        dtype=np.float32,
    )

    comparison_rows: List[
        Dict[str, Any]
    ] = []

    print("\nGörseller oluşturuluyor:")

    for config_id in CANDIDATE_CONFIG_IDS:
        labels = np.array(
            [
                row[f"{config_id}_label"]
                for row in hdbscan_rows
            ],
            dtype=np.int32,
        )

        probabilities = np.array(
            [
                row[
                    f"{config_id}_probability"
                ]
                for row in hdbscan_rows
            ],
            dtype=np.float32,
        )

        plot_path = create_candidate_plot(
            config_id=config_id,
            points_2d=points_2d,
            labels=labels,
            summary=summaries[
                config_id
            ],
        )

        print(
            f"- {config_id}: "
            f"{plot_path}"
        )

        comparison_row = (
            calculate_candidate_comparison(
                config_id=config_id,
                labels=labels,
                probabilities=probabilities,
                kmeans_silhouettes=(
                    kmeans_silhouettes
                ),
                summary=summaries[
                    config_id
                ],
            )
        )

        comparison_rows.append(
            comparison_row
        )

    print_comparison(
        comparison_rows
    )

    comparison_path = (
        save_comparison_csv(
            comparison_rows
        )
    )

    print("\n" + "=" * 80)
    print("DAY 18 TAMAMLANDI")
    print("=" * 80)

    print(
        f"\nAday karşılaştırma CSV:\n"
        f"{comparison_path}"
    )

    print(
        "\nOluşan üç görselde özellikle şunlara bak:\n"
        "1. Clusterlar ayrı ve tutarlı adacıklar oluşturuyor mu?\n"
        "2. Noise noktaları geçiş bölgelerinde mi?\n"
        "3. H18 önemli konu bölgelerini fazla mı siliyor?"
    )


if __name__ == "__main__":
    main()