import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    import umap
except ImportError as error:
    raise RuntimeError(
        "umap-learn kurulu değil.\n"
        "Şu komutu çalıştır:\n"
        "python -m pip install "
        "\"numpy<1.27\" "
        "\"numba==0.58.1\" "
        "\"pynndescent==0.5.10\" "
        "\"umap-learn==0.5.5\""
    ) from error


# ---------------------------------------------------------
# 1. UMAP ayarları
# ---------------------------------------------------------
#
# n_neighbors=15:
# Her makalenin çevresindeki yaklaşık 15 komşuya bakar.
# Yerel konu ilişkilerini görmemiz için başlangıç değeridir.
#
# min_dist=0.1:
# Benzer noktaların 2D görünümde nispeten sıkı
# kümelenmesine izin verir.
#
# metric="cosine":
# Embedding karşılaştırmalarında kullandığımız
# cosine uzaklığına uygun hareket eder.
#
# Bu değerler nihai veya tek doğru değerler değildir.
#

N_NEIGHBORS = 15
MIN_DIST = 0.1
RANDOM_SEED = 42


# ---------------------------------------------------------
# 2. Geçici insan yorumlu cluster adları
# ---------------------------------------------------------
#
# Bunlar modelin ürettiği resmi etiketler değildir.
# Day 15 raporundaki temsilci başlıklar, keywords ve
# subject dağılımları incelenerek verilen taslak adlardır.
#
# Düşük güven:
# Cluster sınırı karışık veya ortalama silhouette negatif.
#
# Orta güven:
# Genel konu anlaşılabiliyor fakat alt alanlar karışabiliyor.
#
# Yüksek güven:
# Temsilci makaleler ve subject dağılımı oldukça tutarlı.
#

CLUSTER_LABELS: Dict[int, Tuple[str, str]] = {
    0: (
        "Yer Bilimleri, Jeoloji ve Doğal Afetler",
        "orta",
    ),
    1: (
        "Sosyoekonomik Kalkınma ve Kamu Politikaları",
        "düşük",
    ),
    2: (
        "Anayasa, Demokrasi ve Siyasal-Hukuki Düşünce",
        "orta",
    ),
    3: (
        "İletişim, Medya ve Halkla İlişkiler",
        "yüksek",
    ),
    4: (
        "Enfeksiyon, Mikrobiyoloji ve Parazitoloji",
        "yüksek",
    ),
    5: (
        "Gebelik, Doğum ve Perinatal Sağlık",
        "orta",
    ),
    6: (
        "Eğitim Araştırmaları ve Eğitim Teknolojileri",
        "orta",
    ),
    7: (
        "Osmanlı ve Yakın Dönem Siyasal-Hukuki Tarihi",
        "düşük",
    ),
    8: (
        "Onkoloji ve Tümör Patolojisi",
        "yüksek",
    ),
    9: (
        "Şirketler ve Ticaret Hukuku",
        "yüksek",
    ),
    10: (
        "Klasik Türk Edebiyatı ve Metin İncelemeleri",
        "yüksek",
    ),
    11: (
        "Bölgesel Tarih, Tarihî Coğrafya ve Arkeoloji",
        "orta",
    ),
    12: (
        "Finansal Piyasalar ve Ekonomik Büyüme",
        "yüksek",
    ),
    13: (
        "Akut Klinik Tıp ve Cerrahi Uygulamalar",
        "düşük",
    ),
    14: (
        "Turizm, Konaklama ve Sürdürülebilir Turizm",
        "yüksek",
    ),
    15: (
        "Psikoloji, Aile ve Psikososyal Sağlık",
        "orta",
    ),
    16: (
        "Mühendislik Tasarımı, İmalat ve Kalite",
        "düşük",
    ),
    17: (
        "Vergi, İdare ve Düzenleyici Hukuk",
        "düşük",
    ),
    18: (
        "Gıda Bilimi, Güvenliği ve Mikrobiyoloji",
        "yüksek",
    ),
    19: (
        "Modern Türk Edebiyatı ve Edebi Eleştiri",
        "yüksek",
    ),
    20: (
        "Çevre ve Halk Sağlığı",
        "düşük",
    ),
    21: (
        "Sinema, Sanat, Kültürel Temsil ve Bellek",
        "orta",
    ),
    22: (
        "Felsefe, Din Bilimleri ve Hermenötik",
        "yüksek",
    ),
    23: (
        "Sağlık Hizmetleri, Hemşirelik ve Dijital Sağlık",
        "orta",
    ),
    24: (
        "Bitki Bilimleri ve Uygulamalı Biyoloji",
        "yüksek",
    ),
    25: (
        "Göz Hastalıkları ve Oftalmoloji",
        "yüksek",
    ),
    26: (
        "Ceza ve Usul Hukuku",
        "orta",
    ),
    27: (
        "Ortopedi, Romatoloji ve Rehabilitasyon",
        "yüksek",
    ),
    28: (
        "Türk Dili, Ağızlar ve Kültürdilbilim",
        "yüksek",
    ),
    29: (
        "E-Ticaret, Dijital Pazarlama ve Tüketici Davranışı",
        "orta",
    ),
}


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


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

    if embeddings.shape[0] != 1000:
        raise ValueError(
            "1.000 embedding satırı bekleniyordu, "
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

    # Day 13'te normalize edildi ama burada yeniden
    # doğrulayıp güvenli biçimde normalize ediyoruz.
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


def load_assignments() -> List[Dict[str, Any]]:
    """
    Day 15 KMeans atamalarını okur.

    UMAP yeniden cluster üretmez.
    Buradaki mevcut cluster_id değerlerini kullanır.
    """

    input_path = (
        get_project_root()
        / "research" / "outputs"
        / "day15_tr_mteb_k30_assignments.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Cluster atama dosyası bulunamadı:\n{input_path}"
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
            "publication_year",
            "title_tr",
            "keywords_tr",
            "subjects",
        }

        existing_columns = set(
            reader.fieldnames or []
        )

        missing_columns = (
            required_columns - existing_columns
        )

        if missing_columns:
            raise ValueError(
                "Atama CSV'sinde eksik sütunlar var: "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            rows.append(
                {
                    "row_index": int(row["row_index"]),
                    "article_id": row["article_id"],
                    "cluster_id": int(row["cluster_id"]),
                    "silhouette": float(row["silhouette"]),
                    "publication_year": row[
                        "publication_year"
                    ],
                    "title_tr": row["title_tr"],
                    "keywords_tr": row["keywords_tr"],
                    "subjects": row["subjects"],
                }
            )

    rows.sort(
        key=lambda row: row["row_index"]
    )

    if len(rows) != 1000:
        raise ValueError(
            "1.000 cluster ataması bekleniyordu, "
            f"bulunan: {len(rows)}"
        )

    expected_indices = list(range(len(rows)))

    actual_indices = [
        row["row_index"]
        for row in rows
    ]

    if actual_indices != expected_indices:
        raise ValueError(
            "row_index değerleri 0–999 sırasında değil."
        )

    return rows


def run_umap(
    embeddings: np.ndarray,
) -> np.ndarray:
    """768 boyutlu vektörleri iki boyuta indirir."""

    print("\nUMAP çalıştırılıyor...")
    print(f"- n_neighbors: {N_NEIGHBORS}")
    print(f"- min_dist   : {MIN_DIST}")
    print("- metric     : cosine")
    print("- bileşen    : 2")

    reducer = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        n_components=2,
        metric="cosine",
        random_state=RANDOM_SEED,
        low_memory=True,
    )

    points_2d = reducer.fit_transform(
        embeddings
    )

    if points_2d.shape != (1000, 2):
        raise ValueError(
            f"Beklenmeyen UMAP şekli: {points_2d.shape}"
        )

    return points_2d.astype(
        np.float32,
        copy=False,
    )


def save_provisional_labels() -> Path:
    """Geçici insan yorumlu cluster adlarını CSV'ye yazar."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day16_provisional_cluster_labels.csv"
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "cluster_id",
                "provisional_topic_label",
                "confidence",
                "status",
            ],
        )

        writer.writeheader()

        for cluster_id in sorted(
            CLUSTER_LABELS
        ):
            topic_label, confidence = (
                CLUSTER_LABELS[cluster_id]
            )

            writer.writerow(
                {
                    "cluster_id": cluster_id,
                    "provisional_topic_label": (
                        topic_label
                    ),
                    "confidence": confidence,
                    "status": (
                        "İnsan yorumu; ground truth değildir"
                    ),
                }
            )

    return output_path


def save_coordinates(
    points_2d: np.ndarray,
    assignments: List[Dict[str, Any]],
) -> Path:
    """UMAP koordinatlarını makale bilgileriyle CSV'ye yazar."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day16_tr_mteb_umap_coordinates.csv"
    )

    fieldnames = [
        "row_index",
        "article_id",
        "umap_x",
        "umap_y",
        "cluster_id",
        "provisional_topic_label",
        "label_confidence",
        "silhouette",
        "publication_year",
        "title_tr",
        "keywords_tr",
        "subjects",
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

        for row, point in zip(
            assignments,
            points_2d,
        ):
            cluster_id = row["cluster_id"]

            topic_label, confidence = (
                CLUSTER_LABELS[cluster_id]
            )

            writer.writerow(
                {
                    "row_index": row["row_index"],
                    "article_id": row["article_id"],
                    "umap_x": float(point[0]),
                    "umap_y": float(point[1]),
                    "cluster_id": cluster_id,
                    "provisional_topic_label": (
                        topic_label
                    ),
                    "label_confidence": confidence,
                    "silhouette": row["silhouette"],
                    "publication_year": row[
                        "publication_year"
                    ],
                    "title_tr": row["title_tr"],
                    "keywords_tr": row[
                        "keywords_tr"
                    ],
                    "subjects": row["subjects"],
                }
            )

    return output_path


def calculate_cluster_label_positions(
    points_2d: np.ndarray,
    cluster_ids: np.ndarray,
) -> Dict[int, Tuple[float, float]]:
    """
    Her cluster numarasının yazılacağı 2D konumu hesaplar.

    Ortalama yerine medyan kullanmak uç noktaların
    yazıyı fazla kaydırmasını azaltır.
    """

    positions: Dict[
        int,
        Tuple[float, float],
    ] = {}

    for cluster_id in sorted(
        set(int(value) for value in cluster_ids)
    ):
        cluster_points = points_2d[
            cluster_ids == cluster_id
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


def create_cluster_plot(
    points_2d: np.ndarray,
    cluster_ids: np.ndarray,
) -> Path:
    """
    Makaleleri KMeans cluster numarasına göre gösterir.

    Renkler UMAP tarafından oluşturulmaz.
    Renk, Day 15'teki mevcut KMeans cluster ID'sidir.
    """

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day16_tr_mteb_umap_clusters.png"
    )

    label_positions = (
        calculate_cluster_label_positions(
            points_2d=points_2d,
            cluster_ids=cluster_ids,
        )
    )

    plt.figure(figsize=(16, 11))

    scatter = plt.scatter(
        points_2d[:, 0],
        points_2d[:, 1],
        c=cluster_ids,
        s=28,
        alpha=0.75,
    )

    colorbar = plt.colorbar(scatter)
    colorbar.set_label("KMeans Cluster ID")

    for cluster_id, (
        x_position,
        y_position,
    ) in label_positions.items():
        plt.annotate(
            str(cluster_id),
            (x_position, y_position),
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.2",
                "alpha": 0.75,
            },
        )

    plt.title(
        "TR-MTEB Embeddinglerinin UMAP ile 2 Boyutlu Gösterimi\n"
        "Renkler Day 15 KMeans k=30 cluster atamalarıdır"
    )

    plt.xlabel("UMAP Boyutu 1")
    plt.ylabel("UMAP Boyutu 2")

    plt.figtext(
        0.5,
        0.015,
        (
            "UMAP eksenlerinin tek başına konu anlamı yoktur. "
            "Yakın noktalar benzer yerel komşuluk yapısına sahiptir."
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


def create_silhouette_plot(
    points_2d: np.ndarray,
    silhouettes: np.ndarray,
) -> Path:
    """
    Aynı UMAP koordinatlarını silhouette değerine göre gösterir.

    Düşük veya negatif değerler:
    - cluster sınırındaki makaleleri,
    - başka clusterlara yakın makaleleri,
    - çok alanlı veya belirsiz örnekleri

    işaret edebilir.
    """

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day16_tr_mteb_umap_silhouette.png"
    )

    negative_count = int(
        np.sum(silhouettes < 0)
    )

    plt.figure(figsize=(15, 10))

    scatter = plt.scatter(
        points_2d[:, 0],
        points_2d[:, 1],
        c=silhouettes,
        s=30,
        alpha=0.8,
    )

    colorbar = plt.colorbar(scatter)

    colorbar.set_label(
        "Makale bazlı cosine silhouette"
    )

    plt.title(
        "TR-MTEB UMAP Görünümünde Cluster Sınırları\n"
        f"Negatif silhouette değerine sahip makale: "
        f"{negative_count}/1000"
    )

    plt.xlabel("UMAP Boyutu 1")
    plt.ylabel("UMAP Boyutu 2")

    plt.figtext(
        0.5,
        0.015,
        (
            "Silhouette düşükse makale kendi clusterına "
            "keskin biçimde bağlı olmayabilir."
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


def print_summary(
    points_2d: np.ndarray,
    cluster_ids: np.ndarray,
    silhouettes: np.ndarray,
) -> None:
    """Terminale kısa teknik özet yazar."""

    print("\n" + "=" * 80)
    print("UMAP TEKNİK ÖZETİ")
    print("=" * 80)

    print(f"\nKoordinat şekli : {points_2d.shape}")
    print(
        f"Cluster sayısı  : "
        f"{len(np.unique(cluster_ids))}"
    )
    print(
        f"Negatif silhouette makale sayısı: "
        f"{int(np.sum(silhouettes < 0))}"
    )
    print(
        f"Ortalama silhouette: "
        f"{float(np.mean(silhouettes)):.4f}"
    )
    print(
        f"UMAP X aralığı  : "
        f"{points_2d[:, 0].min():.3f} – "
        f"{points_2d[:, 0].max():.3f}"
    )
    print(
        f"UMAP Y aralığı  : "
        f"{points_2d[:, 1].min():.3f} – "
        f"{points_2d[:, 1].max():.3f}"
    )


def main() -> None:
    print("=" * 80)
    print("DAY 16 — TR-MTEB UMAP GÖRSELLEŞTİRMESİ")
    print("=" * 80)

    embeddings = load_embeddings()
    assignments = load_assignments()

    cluster_ids = np.array(
        [
            row["cluster_id"]
            for row in assignments
        ],
        dtype=np.int32,
    )

    silhouettes = np.array(
        [
            row["silhouette"]
            for row in assignments
        ],
        dtype=np.float32,
    )

    points_2d = run_umap(
        embeddings=embeddings
    )

    label_path = save_provisional_labels()

    coordinate_path = save_coordinates(
        points_2d=points_2d,
        assignments=assignments,
    )

    cluster_plot_path = create_cluster_plot(
        points_2d=points_2d,
        cluster_ids=cluster_ids,
    )

    silhouette_plot_path = (
        create_silhouette_plot(
            points_2d=points_2d,
            silhouettes=silhouettes,
        )
    )

    print_summary(
        points_2d=points_2d,
        cluster_ids=cluster_ids,
        silhouettes=silhouettes,
    )

    print("\n" + "=" * 80)
    print("DAY 16 TAMAMLANDI")
    print("=" * 80)

    print(
        f"\nGeçici konu etiketleri:\n"
        f"{label_path}"
    )

    print(
        f"\nMakale bazlı UMAP koordinatları:\n"
        f"{coordinate_path}"
    )

    print(
        f"\nCluster UMAP görseli:\n"
        f"{cluster_plot_path}"
    )

    print(
        f"\nSilhouette UMAP görseli:\n"
        f"{silhouette_plot_path}"
    )


if __name__ == "__main__":
    main()