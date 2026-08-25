import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


MODEL_FILES = {
    "TR-MTEB": "tr_mteb.npy",
    "E5-large": "e5_large.npy",
    "GTE-multilingual": "gte_multilingual.npy",
}

K_VALUES = [
    5,
    10,
    15,
    20,
    30,
    40,
    50,
]

RANDOM_SEED = 42
N_INIT = 20


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


def load_embeddings() -> Dict[str, np.ndarray]:
    """
    Day 13 aşamasında oluşturulan embedding
    matrislerini yükler ve doğrular.
    """

    embedding_directory = (
        get_project_root()
        / "research" / "outputs"
        / "day13_embeddings"
    )

    loaded_embeddings: Dict[str, np.ndarray] = {}

    print("=" * 80)
    print("EMBEDDING DOSYALARINI DOĞRULAMA")
    print("=" * 80)

    for model_name, filename in MODEL_FILES.items():
        embedding_path = (
            embedding_directory
            / filename
        )

        if not embedding_path.exists():
            raise FileNotFoundError(
                f"{model_name} dosyası bulunamadı:\n"
                f"{embedding_path}"
            )

        embeddings = np.load(
            embedding_path
        )

        if embeddings.ndim != 2:
            raise ValueError(
                f"{model_name} matrisi iki boyutlu değil: "
                f"{embeddings.shape}"
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                f"{model_name} matrisinde NaN veya "
                f"sonsuz değer var."
            )

        if embeddings.shape[0] != 1000:
            raise ValueError(
                f"{model_name} için 1.000 satır bekleniyordu, "
                f"bulunan: {embeddings.shape[0]}"
            )

        embeddings = embeddings.astype(
            np.float32,
            copy=False,
        )

        norms = np.linalg.norm(
            embeddings,
            axis=1,
        )

        unique_vector_count = np.unique(
            embeddings,
            axis=0,
        ).shape[0]

        print(f"\nModel              : {model_name}")
        print(f"Matris şekli       : {embeddings.shape}")
        print(f"Veri tipi          : {embeddings.dtype}")
        print(f"Ortalama norm       : {norms.mean():.6f}")
        print(f"Minimum norm        : {norms.min():.6f}")
        print(f"Maksimum norm       : {norms.max():.6f}")
        print(
            f"Benzersiz vektör    : "
            f"{unique_vector_count}/{embeddings.shape[0]}"
        )

        loaded_embeddings[
            model_name
        ] = embeddings

    return loaded_embeddings


def calculate_cluster_statistics(
    labels: np.ndarray,
) -> Dict[str, float]:
    """Cluster büyüklüklerinin temel istatistiklerini hesaplar."""

    cluster_counter = Counter(
        int(label)
        for label in labels
    )

    cluster_sizes = np.array(
        list(cluster_counter.values()),
        dtype=np.float64,
    )

    return {
        "minimum_cluster_size": int(
            cluster_sizes.min()
        ),
        "maximum_cluster_size": int(
            cluster_sizes.max()
        ),
        "mean_cluster_size": float(
            cluster_sizes.mean()
        ),
        "median_cluster_size": float(
            np.median(cluster_sizes)
        ),
        "cluster_size_std": float(
            cluster_sizes.std()
        ),
    }


def run_kmeans_experiment(
    model_name: str,
    embeddings: np.ndarray,
    k: int,
) -> Dict[str, Any]:
    """Tek model ve tek k değeri için KMeans çalıştırır."""

    print(
        f"Çalışıyor: model={model_name}, k={k}"
    )

    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_SEED,
        n_init=N_INIT,
    )

    labels = kmeans.fit_predict(
        embeddings
    )

    # Clustering yüksek boyutlu embeddingler üzerinde yapılır.
    # Cosine silhouette yalnızca kaliteyi değerlendirmek için kullanılır.
    cosine_silhouette = silhouette_score(
        embeddings,
        labels,
        metric="cosine",
    )

    euclidean_silhouette = silhouette_score(
        embeddings,
        labels,
        metric="euclidean",
    )

    calinski_harabasz = (
        calinski_harabasz_score(
            embeddings,
            labels,
        )
    )

    davies_bouldin = (
        davies_bouldin_score(
            embeddings,
            labels,
        )
    )

    cluster_statistics = (
        calculate_cluster_statistics(
            labels
        )
    )

    result: Dict[str, Any] = {
        "model_name": model_name,
        "k": k,
        "article_count": int(
            embeddings.shape[0]
        ),
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "cosine_silhouette": float(
            cosine_silhouette
        ),
        "euclidean_silhouette": float(
            euclidean_silhouette
        ),
        "calinski_harabasz": float(
            calinski_harabasz
        ),
        "davies_bouldin": float(
            davies_bouldin
        ),
        "inertia": float(
            kmeans.inertia_
        ),
        **cluster_statistics,
    }

    return result


def run_all_experiments(
    model_embeddings: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    """Bütün model ve k kombinasyonlarını çalıştırır."""

    print("\n" + "=" * 80)
    print("KMEANS K DEĞERİ TARAMASI")
    print("=" * 80)

    results: List[Dict[str, Any]] = []

    for model_name, embeddings in (
        model_embeddings.items()
    ):
        print("\n" + "-" * 80)
        print(f"MODEL: {model_name}")
        print("-" * 80)

        for k in K_VALUES:
            result = run_kmeans_experiment(
                model_name=model_name,
                embeddings=embeddings,
                k=k,
            )

            results.append(result)

            print(
                f"  cosine silhouette="
                f"{result['cosine_silhouette']:.4f} "
                f"| min={result['minimum_cluster_size']} "
                f"| max={result['maximum_cluster_size']}"
            )

    return results


def print_best_results(
    results: List[Dict[str, Any]],
) -> None:
    """Her model için en yüksek cosine silhouette sonucunu gösterir."""

    print("\n" + "=" * 80)
    print("HER MODEL İÇİN EN YÜKSEK COSINE SILHOUETTE")
    print("=" * 80)

    model_names = sorted(
        {
            result["model_name"]
            for result in results
        }
    )

    for model_name in model_names:
        model_results = [
            result
            for result in results
            if result["model_name"] == model_name
        ]

        best_result = max(
            model_results,
            key=lambda result: result[
                "cosine_silhouette"
            ],
        )

        print(f"\nModel       : {model_name}")
        print(f"En iyi k    : {best_result['k']}")
        print(
            f"Cosine silhouette: "
            f"{best_result['cosine_silhouette']:.4f}"
        )
        print(
            f"Cluster boyutları : "
            f"min={best_result['minimum_cluster_size']}, "
            f"max={best_result['maximum_cluster_size']}, "
            f"medyan={best_result['median_cluster_size']:.1f}"
        )

        print(
            "\nNot: Bu sonuç tek başına nihai k seçimi değildir."
        )


def save_results(
    results: List[Dict[str, Any]],
) -> Tuple[Path, Path]:

    output_directory = (
        get_project_root()
        / "research" / "outputs"
    )

    csv_path = (
        output_directory
        / "day14_kmeans_sweep.csv"
    )

    json_path = (
        output_directory
        / "day14_kmeans_sweep.json"
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
        writer.writerows(
            results
        )

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


def create_silhouette_chart(
    results: List[Dict[str, Any]],
) -> Path:
    """K değerine göre cosine silhouette grafiği oluşturur."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day14_kmeans_silhouette.png"
    )

    plt.figure(
        figsize=(11, 7)
    )

    model_names = sorted(
        {
            result["model_name"]
            for result in results
        }
    )

    for model_name in model_names:
        model_results = sorted(
            [
                result
                for result in results
                if result["model_name"]
                == model_name
            ],
            key=lambda result: result["k"],
        )

        k_values = [
            result["k"]
            for result in model_results
        ]

        silhouette_values = [
            result["cosine_silhouette"]
            for result in model_results
        ]

        plt.plot(
            k_values,
            silhouette_values,
            marker="o",
            label=model_name,
        )

    plt.title(
        "KMeans: k Değerine Göre Cosine Silhouette"
    )
    plt.xlabel("Cluster sayısı (k)")
    plt.ylabel("Cosine silhouette")
    plt.xticks(K_VALUES)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=160,
    )
    plt.close()

    return output_path


def main() -> None:
    model_embeddings = (
        load_embeddings()
    )

    results = run_all_experiments(
        model_embeddings=model_embeddings
    )

    print_best_results(
        results=results
    )

    csv_path, json_path = save_results(
        results=results
    )

    chart_path = create_silhouette_chart(
        results=results
    )

    print("\n" + "=" * 80)
    print("DAY 14 TAMAMLANDI")
    print("=" * 80)

    print(f"\nSonuç CSV:\n{csv_path}")
    print(f"\nSonuç JSON:\n{json_path}")
    print(f"\nSilhouette görseli:\n{chart_path}")


if __name__ == "__main__":
    main()