import csv
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import umap
import hdbscan
from sklearn.metrics import silhouette_score


# =========================================================
# 1. UMAP AYARLARI
# =========================================================
#
# Day 16:
# 768D -> 2D yalnızca görselleştirme içindi.
#
# Day 17:
# 768D -> 10D HDBSCAN girdisi olacak.
#
# 10 boyutun konu anlamı yoktur. Ama 2 boyuta göre
# daha fazla komşuluk bilgisini korumaya çalışır.
#

UMAP_COMPONENTS = 10
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0
RANDOM_SEED = 42


# =========================================================
# 2. HDBSCAN PARAMETRE TARAMASI
# =========================================================

MIN_CLUSTER_SIZES = [
    10,
    15,
    20,
    30,
    40,
]

MIN_SAMPLES_VALUES = [
    5,
    10,
    15,
]

CLUSTER_SELECTION_METHODS = [
    "eom",
    "leaf",
]


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


def load_embeddings() -> np.ndarray:
    """
    Day 13'te üretilen TR-MTEB embeddinglerini yükler.

    Embedding sırası pilot_articles_1000.jsonl ve
    Day 15 assignments dosyasıyla aynıdır.
    """

    input_path = (
        get_project_root()
        / "research" / "outputs"
        / "day13_embeddings"
        / "tr_mteb.npy"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"TR-MTEB embedding dosyası bulunamadı:\n"
            f"{input_path}"
        )

    embeddings = np.load(input_path)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embedding matrisi iki boyutlu değil: "
            f"{embeddings.shape}"
        )

    if embeddings.shape[0] != 1000:
        raise ValueError(
            "1.000 embedding bekleniyordu, "
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

    # Embeddingler Day 13'te zaten normalize edildi.
    # Burada güvenli biçimde tekrar doğruluyoruz.
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
    """
    Day 15 KMeans sonuçlarını okur.

    Özellikle KMeans silhouette değeri negatif olan
    204 makalenin HDBSCAN tarafından nasıl ele
    alındığını karşılaştıracağız.
    """

    input_path = (
        get_project_root()
        / "research" / "outputs"
        / "day15_tr_mteb_k30_assignments.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"KMeans atama dosyası bulunamadı:\n"
            f"{input_path}"
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
                "Day 15 CSV'sinde eksik sütunlar var: "
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
                    "kmeans_cluster_id": int(
                        row["cluster_id"]
                    ),
                    "kmeans_silhouette": float(
                        row["silhouette"]
                    ),
                    "title_tr": row[
                        "title_tr"
                    ],
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

    expected_indices = list(
        range(1000)
    )

    actual_indices = [
        row["row_index"]
        for row in rows
    ]

    if actual_indices != expected_indices:
        raise ValueError(
            "row_index değerleri 0–999 sırasında değil."
        )

    return rows


def reduce_embeddings_to_10d(
    embeddings: np.ndarray,
) -> np.ndarray:
    """TR-MTEB embeddinglerini UMAP ile 10 boyuta indirir."""

    print("\n" + "=" * 80)
    print("UMAP İLE HDBSCAN GİRDİSİ HAZIRLAMA")
    print("=" * 80)

    print(f"\nGirdi şekli   : {embeddings.shape}")
    print(
        f"Çıktı boyutu  : "
        f"{UMAP_COMPONENTS}"
    )
    print(
        f"n_neighbors   : "
        f"{UMAP_N_NEIGHBORS}"
    )
    print(
        f"min_dist      : "
        f"{UMAP_MIN_DIST}"
    )
    print("metric        : cosine")

    reducer = umap.UMAP(
        n_components=UMAP_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=RANDOM_SEED,
        low_memory=True,
    )

    reduced_embeddings = (
        reducer.fit_transform(
            embeddings
        )
    )

    expected_shape = (
        embeddings.shape[0],
        UMAP_COMPONENTS,
    )

    if reduced_embeddings.shape != expected_shape:
        raise ValueError(
            f"Beklenmeyen UMAP şekli: "
            f"{reduced_embeddings.shape}"
        )

    reduced_embeddings = (
        reduced_embeddings.astype(
            np.float32,
            copy=False,
        )
    )

    print(
        f"Üretilen şekil: "
        f"{reduced_embeddings.shape}"
    )

    return reduced_embeddings


def save_reduced_embeddings(
    reduced_embeddings: np.ndarray,
) -> Path:
    """HDBSCAN için hazırlanan 10D UMAP verisini kaydeder."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day17_tr_mteb_umap_10d.npy"
    )

    np.save(
        output_path,
        reduced_embeddings,
    )

    return output_path


def build_parameter_configs() -> List[Dict[str, Any]]:
    """Deneceğimiz HDBSCAN parametrelerini oluşturur."""

    configs: List[Dict[str, Any]] = []

    config_number = 1

    for selection_method in (
        CLUSTER_SELECTION_METHODS
    ):
        for min_cluster_size in (
            MIN_CLUSTER_SIZES
        ):
            for min_samples in (
                MIN_SAMPLES_VALUES
            ):
                # min_samples değerinin minimum cluster
                # boyutundan büyük olmasını bu ilk deneyde
                # gereksiz yere aşırı katı buluyoruz.
                if min_samples > min_cluster_size:
                    continue

                configs.append(
                    {
                        "config_id": (
                            f"H{config_number:02d}"
                        ),
                        "min_cluster_size": (
                            min_cluster_size
                        ),
                        "min_samples": (
                            min_samples
                        ),
                        "selection_method": (
                            selection_method
                        ),
                    }
                )

                config_number += 1

    return configs


def calculate_cluster_size_statistics(
    labels: np.ndarray,
) -> Dict[str, Any]:
    """Noise dışındaki cluster büyüklüklerini hesaplar."""

    non_noise_labels = labels[
        labels >= 0
    ]

    if non_noise_labels.size == 0:
        return {
            "minimum_cluster_size_found": 0,
            "maximum_cluster_size_found": 0,
            "mean_cluster_size_found": 0.0,
            "median_cluster_size_found": 0.0,
        }

    unique_labels, counts = np.unique(
        non_noise_labels,
        return_counts=True,
    )

    del unique_labels

    return {
        "minimum_cluster_size_found": int(
            counts.min()
        ),
        "maximum_cluster_size_found": int(
            counts.max()
        ),
        "mean_cluster_size_found": float(
            counts.mean()
        ),
        "median_cluster_size_found": float(
            median(
                [
                    int(count)
                    for count in counts
                ]
            )
        ),
    }


def calculate_silhouette_safely(
    data: np.ndarray,
    labels: np.ndarray,
    metric: str,
) -> Optional[float]:
    """
    Noise noktalarını hariç tutarak silhouette hesaplar.

    HDBSCAN yalnızca tek cluster üretirse silhouette
    hesaplanamaz ve None döner.
    """

    clustered_mask = labels >= 0

    clustered_labels = labels[
        clustered_mask
    ]

    unique_clusters = np.unique(
        clustered_labels
    )

    if len(unique_clusters) < 2:
        return None

    if clustered_labels.size <= len(
        unique_clusters
    ):
        return None

    return float(
        silhouette_score(
            data[clustered_mask],
            clustered_labels,
            metric=metric,
        )
    )


def safe_float(
    value: Any,
) -> Optional[float]:
    """NaN veya sonsuz değerleri JSON için None yapar."""

    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(float_value):
        return None

    return float_value


def evaluate_configuration(
    config: Dict[str, Any],
    original_embeddings: np.ndarray,
    reduced_embeddings: np.ndarray,
    kmeans_silhouettes: np.ndarray,
) -> Tuple[
    Dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Tek bir HDBSCAN yapılandırmasını çalıştırır."""

    print(
        f"\n{config['config_id']} "
        f"| method={config['selection_method']} "
        f"| min_cluster_size="
        f"{config['min_cluster_size']} "
        f"| min_samples="
        f"{config['min_samples']}"
    )

    clusterer = hdbscan.HDBSCAN(
    min_cluster_size=config[
        "min_cluster_size"
    ],
    min_samples=config[
        "min_samples"
    ],
    metric="euclidean",
    cluster_selection_method=config[
        "selection_method"
    ],
    prediction_data=True,

    # relative_validity_ metriğini kullanmadığımız için
    # minimum spanning tree nesnesini ayrıca saklamıyoruz.
    gen_min_span_tree=False,
)

    labels = clusterer.fit_predict(
        reduced_embeddings
    )

    probabilities = (
        clusterer.probabilities_
        .astype(
            np.float32,
            copy=False,
        )
    )

    outlier_scores = (
        clusterer.outlier_scores_
        .astype(
            np.float32,
            copy=False,
        )
    )

    unique_cluster_labels = {
        int(label)
        for label in labels
        if int(label) >= 0
    }

    cluster_count = len(
        unique_cluster_labels
    )

    noise_mask = labels == -1
    clustered_mask = labels >= 0

    noise_count = int(
        np.sum(noise_mask)
    )

    clustered_count = int(
        np.sum(clustered_mask)
    )

    noise_rate = (
        noise_count
        / len(labels)
    )

    negative_kmeans_mask = (
        kmeans_silhouettes < 0
    )

    positive_kmeans_mask = (
        ~negative_kmeans_mask
    )

    negative_kmeans_count = int(
        np.sum(
            negative_kmeans_mask
        )
    )

    captured_negative_count = int(
        np.sum(
            noise_mask
            & negative_kmeans_mask
        )
    )

    negative_capture_rate = (
        captured_negative_count
        / negative_kmeans_count
        if negative_kmeans_count
        else 0.0
    )

    noise_negative_share = (
        captured_negative_count
        / noise_count
        if noise_count
        else 0.0
    )

    positive_kmeans_count = int(
        np.sum(
            positive_kmeans_mask
        )
    )

    positive_kmeans_noise_count = int(
        np.sum(
            noise_mask
            & positive_kmeans_mask
        )
    )

    positive_kmeans_noise_rate = (
        positive_kmeans_noise_count
        / positive_kmeans_count
        if positive_kmeans_count
        else 0.0
    )

    original_cosine_silhouette = (
        calculate_silhouette_safely(
            data=original_embeddings,
            labels=labels,
            metric="cosine",
        )
    )

    reduced_euclidean_silhouette = (
        calculate_silhouette_safely(
            data=reduced_embeddings,
            labels=labels,
            metric="euclidean",
        )
    )

    if clustered_count:
        mean_membership_probability = float(
            probabilities[
                clustered_mask
            ].mean()
        )

        median_membership_probability = float(
            np.median(
                probabilities[
                    clustered_mask
                ]
            )
        )
    else:
        mean_membership_probability = 0.0
        median_membership_probability = 0.0

    relative_validity = None

    cluster_statistics = (
        calculate_cluster_size_statistics(
            labels
        )
    )

    result: Dict[str, Any] = {
        "config_id": config[
            "config_id"
        ],
        "selection_method": config[
            "selection_method"
        ],
        "min_cluster_size": config[
            "min_cluster_size"
        ],
        "min_samples": config[
            "min_samples"
        ],
        "cluster_count": cluster_count,
        "clustered_article_count": (
            clustered_count
        ),
        "noise_count": noise_count,
        "noise_rate": float(
            noise_rate
        ),
        "mean_membership_probability": (
            mean_membership_probability
        ),
        "median_membership_probability": (
            median_membership_probability
        ),
        "original_cosine_silhouette": (
            original_cosine_silhouette
        ),
        "reduced_euclidean_silhouette": (
            reduced_euclidean_silhouette
        ),
        "relative_validity": (
            relative_validity
        ),
        "negative_kmeans_article_count": (
            negative_kmeans_count
        ),
        "captured_negative_count": (
            captured_negative_count
        ),
        "negative_capture_rate": float(
            negative_capture_rate
        ),
        "noise_negative_share": float(
            noise_negative_share
        ),
        "positive_kmeans_noise_count": (
            positive_kmeans_noise_count
        ),
        "positive_kmeans_noise_rate": float(
            positive_kmeans_noise_rate
        ),
        **cluster_statistics,
    }

    print(
        f"  Cluster={cluster_count:2} "
        f"| noise={noise_count:3} "
        f"(%{noise_rate * 100:.1f}) "
        f"| original silhouette="
        f"{original_cosine_silhouette}"
    )

    print(
        f"  KMeans negatif yakalama: "
        f"{captured_negative_count}/"
        f"{negative_kmeans_count} "
        f"(%{negative_capture_rate * 100:.1f})"
    )

    return (
        result,
        labels.astype(
            np.int32,
            copy=False,
        ),
        probabilities,
        outlier_scores,
    )


def run_parameter_sweep(
    original_embeddings: np.ndarray,
    reduced_embeddings: np.ndarray,
    assignments: List[Dict[str, Any]],
) -> Tuple[
    List[Dict[str, Any]],
    Dict[str, Dict[str, np.ndarray]],
]:
    """Bütün HDBSCAN yapılandırmalarını çalıştırır."""

    configs = build_parameter_configs()

    kmeans_silhouettes = np.array(
        [
            row["kmeans_silhouette"]
            for row in assignments
        ],
        dtype=np.float32,
    )

    results: List[Dict[str, Any]] = []

    outputs_by_config: Dict[
        str,
        Dict[str, np.ndarray],
    ] = {}

    print("\n" + "=" * 80)
    print("HDBSCAN PARAMETRE TARAMASI")
    print("=" * 80)

    print(
        f"\nToplam yapılandırma: "
        f"{len(configs)}"
    )

    for config in configs:
        (
            result,
            labels,
            probabilities,
            outlier_scores,
        ) = evaluate_configuration(
            config=config,
            original_embeddings=(
                original_embeddings
            ),
            reduced_embeddings=(
                reduced_embeddings
            ),
            kmeans_silhouettes=(
                kmeans_silhouettes
            ),
        )

        results.append(result)

        outputs_by_config[
            config["config_id"]
        ] = {
            "labels": labels,
            "probabilities": probabilities,
            "outlier_scores": (
                outlier_scores
            ),
        }

    return results, outputs_by_config


def format_metric(
    value: Any,
) -> str:
    """Terminal tablosu için güvenli sayı formatı."""

    if value is None:
        return "-"

    return f"{float(value):.4f}"


def select_balanced_candidates(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    İlk inceleme için aşırı çözümleri filtreler.

    Bunlar bilimsel kesin sınırlar değildir.
    Yalnızca:
    - neredeyse her şeyi noise yapan,
    - yalnızca 1-2 cluster bulan,
    - yüzlerce küçük cluster üreten

    ayarları ilk bakışta geri plana iter.
    """

    candidates = [
        result
        for result in results
        if (
            5
            <= result["cluster_count"]
            <= 60
        )
        and (
            0.05
            <= result["noise_rate"]
            <= 0.45
        )
        and (
            result[
                "original_cosine_silhouette"
            ]
            is not None
        )
    ]

    return sorted(
        candidates,
        key=lambda result: (
            result[
                "original_cosine_silhouette"
            ],
            result[
                "mean_membership_probability"
            ],
        ),
        reverse=True,
    )


def print_candidate_summary(
    results: List[Dict[str, Any]],
) -> None:
    """İncelenmeye değer yapılandırmaları gösterir."""

    candidates = (
        select_balanced_candidates(
            results
        )
    )

    print("\n" + "=" * 100)
    print("DENGELİ ADAYLAR — ORIGINAL COSINE SILHOUETTE SIRASI")
    print("=" * 100)

    if not candidates:
        print(
            "\nBelirlenen dengeli aday aralığına "
            "giren yapılandırma olmadı."
        )
        return

    print(
        "\n"
        f"{'ID':5}"
        f"{'Yöntem':9}"
        f"{'MCS':>6}"
        f"{'MS':>5}"
        f"{'Cluster':>9}"
        f"{'Noise':>10}"
        f"{'Silhouette':>13}"
        f"{'Üyelik':>10}"
        f"{'Negatif yakalama':>18}"
    )

    print("-" * 100)

    for result in candidates[:10]:
        print(
            f"{result['config_id']:5}"
            f"{result['selection_method']:9}"
            f"{result['min_cluster_size']:>6}"
            f"{result['min_samples']:>5}"
            f"{result['cluster_count']:>9}"
            f"{result['noise_rate'] * 100:>9.1f}%"
            f"{format_metric(result['original_cosine_silhouette']):>13}"
            f"{result['mean_membership_probability']:>10.4f}"
            f"{result['negative_capture_rate'] * 100:>17.1f}%"
        )

    print(
        "\nBu sıralama nihai kazanan değildir. "
        "Cluster sayısı, noise oranı ve konu "
        "yorumlanabilirliği birlikte incelenecek."
    )


def save_summary_files(
    results: List[Dict[str, Any]],
) -> Tuple[Path, Path]:
    """HDBSCAN tarama sonuçlarını CSV ve JSON'a kaydeder."""

    output_directory = (
        get_project_root()
        / "research" / "outputs"
    )

    csv_path = (
        output_directory
        / "day17_hdbscan_sweep_summary.csv"
    )

    json_path = (
        output_directory
        / "day17_hdbscan_sweep_summary.json"
    )

    fieldnames = list(
        results[0].keys()
    )

    with csv_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    with json_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            results,
            json_file,
            ensure_ascii=False,
            indent=2,
        )

    return csv_path, json_path


def save_all_assignments(
    assignments: List[Dict[str, Any]],
    outputs_by_config: Dict[
        str,
        Dict[str, np.ndarray],
    ],
) -> Path:
    """
    Her makalenin bütün HDBSCAN ayarlarındaki
    label, üyelik ve outlier sonuçlarını kaydeder.
    """

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day17_hdbscan_all_assignments.csv"
    )

    config_ids = sorted(
        outputs_by_config.keys()
    )

    fieldnames = [
        "row_index",
        "article_id",
        "title_tr",
        "kmeans_cluster_id",
        "kmeans_silhouette",
    ]

    for config_id in config_ids:
        fieldnames.extend(
            [
                f"{config_id}_label",
                f"{config_id}_probability",
                f"{config_id}_outlier_score",
            ]
        )

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

        for row_index, assignment in enumerate(
            assignments
        ):
            output_row: Dict[str, Any] = {
                "row_index": row_index,
                "article_id": assignment[
                    "article_id"
                ],
                "title_tr": assignment[
                    "title_tr"
                ],
                "kmeans_cluster_id": assignment[
                    "kmeans_cluster_id"
                ],
                "kmeans_silhouette": assignment[
                    "kmeans_silhouette"
                ],
            }

            for config_id in config_ids:
                config_output = (
                    outputs_by_config[
                        config_id
                    ]
                )

                output_row[
                    f"{config_id}_label"
                ] = int(
                    config_output[
                        "labels"
                    ][row_index]
                )

                output_row[
                    f"{config_id}_probability"
                ] = float(
                    config_output[
                        "probabilities"
                    ][row_index]
                )

                output_row[
                    f"{config_id}_outlier_score"
                ] = float(
                    config_output[
                        "outlier_scores"
                    ][row_index]
                )

            writer.writerow(
                output_row
            )

    return output_path


def create_cluster_noise_chart(
    results: List[Dict[str, Any]],
) -> Path:
    """
    Her yapılandırmanın cluster sayısı ve
    noise oranı ilişkisini gösterir.
    """

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day17_hdbscan_cluster_vs_noise.png"
    )

    plt.figure(
        figsize=(13, 8)
    )

    for result in results:
        x_value = (
            result["noise_rate"]
            * 100
        )

        y_value = result[
            "cluster_count"
        ]

        plt.scatter(
            x_value,
            y_value,
            s=55,
        )

        plt.annotate(
            result["config_id"],
            (x_value, y_value),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
        )

    plt.title(
        "HDBSCAN Parametreleri: Noise Oranı ve Cluster Sayısı"
    )

    plt.xlabel(
        "Noise olarak bırakılan makale oranı (%)"
    )

    plt.ylabel(
        "Bulunan cluster sayısı"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=170,
    )

    plt.close()

    return output_path


def create_quality_noise_chart(
    results: List[Dict[str, Any]],
) -> Path:
    """
    Original embedding uzayındaki silhouette ile
    noise oranını birlikte gösterir.
    """

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day17_hdbscan_quality_vs_noise.png"
    )

    valid_results = [
        result
        for result in results
        if result[
            "original_cosine_silhouette"
        ]
        is not None
    ]

    plt.figure(
        figsize=(13, 8)
    )

    for result in valid_results:
        x_value = (
            result["noise_rate"]
            * 100
        )

        y_value = result[
            "original_cosine_silhouette"
        ]

        plt.scatter(
            x_value,
            y_value,
            s=55,
        )

        plt.annotate(
            result["config_id"],
            (x_value, y_value),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
        )

    plt.title(
        "HDBSCAN: Noise Oranı ve Original-Uzay "
        "Cosine Silhouette"
    )

    plt.xlabel(
        "Noise olarak bırakılan makale oranı (%)"
    )

    plt.ylabel(
        "Noise hariç original cosine silhouette"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=170,
    )

    plt.close()

    return output_path


def main() -> None:
    print("=" * 80)
    print("DAY 17 — TR-MTEB HDBSCAN PARAMETRE TARAMASI")
    print("=" * 80)

    original_embeddings = (
        load_embeddings()
    )

    assignments = (
        load_kmeans_assignments()
    )

    reduced_embeddings = (
        reduce_embeddings_to_10d(
            embeddings=original_embeddings
        )
    )

    reduced_path = (
        save_reduced_embeddings(
            reduced_embeddings
        )
    )

    (
        results,
        outputs_by_config,
    ) = run_parameter_sweep(
        original_embeddings=(
            original_embeddings
        ),
        reduced_embeddings=(
            reduced_embeddings
        ),
        assignments=assignments,
    )

    print_candidate_summary(
        results
    )

    (
        summary_csv_path,
        summary_json_path,
    ) = save_summary_files(
        results
    )

    assignments_path = (
        save_all_assignments(
            assignments=assignments,
            outputs_by_config=(
                outputs_by_config
            ),
        )
    )

    cluster_noise_chart_path = (
        create_cluster_noise_chart(
            results
        )
    )

    quality_noise_chart_path = (
        create_quality_noise_chart(
            results
        )
    )

    print("\n" + "=" * 80)
    print("DAY 17 TAMAMLANDI")
    print("=" * 80)

    print(
        f"\n10D UMAP verisi:\n"
        f"{reduced_path}"
    )

    print(
        f"\nTarama özet CSV:\n"
        f"{summary_csv_path}"
    )

    print(
        f"\nTarama özet JSON:\n"
        f"{summary_json_path}"
    )

    print(
        f"\nBütün makale atamaları:\n"
        f"{assignments_path}"
    )

    print(
        f"\nCluster/noise grafiği:\n"
        f"{cluster_noise_chart_path}"
    )

    print(
        f"\nKalite/noise grafiği:\n"
        f"{quality_noise_chart_path}"
    )


if __name__ == "__main__":
    main()