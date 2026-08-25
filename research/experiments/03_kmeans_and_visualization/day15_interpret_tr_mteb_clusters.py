import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


# ---------------------------------------------------------
# Deney ayarları
# ---------------------------------------------------------

MODEL_NAME = "TR-MTEB"
EMBEDDING_FILENAME = "tr_mteb.npy"

CLUSTER_COUNT = 30
RANDOM_SEED = 42
N_INIT = 20

REPRESENTATIVE_ARTICLE_COUNT = 5
BOUNDARY_ARTICLE_COUNT = 3
TOP_KEYWORD_COUNT = 12
TOP_SUBJECT_COUNT = 8


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


def load_articles() -> List[Dict[str, Any]]:
    """
    1.000 makalelik pilot JSONL dosyasını okur.

    Makale sırası korunmalıdır. Çünkü embedding matrisindeki
    satırlarla bu listedeki makaleler aynı sıradadır.
    """

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

    if not articles:
        raise ValueError(
            "Pilot veri dosyasında makale bulunamadı."
        )

    return articles


def load_embeddings(
    article_count: int,
) -> np.ndarray:
    """TR-MTEB embedding matrisini yükler ve doğrular."""

    embedding_path = (
        get_project_root()
        / "research" / "outputs"
        / "day13_embeddings"
        / EMBEDDING_FILENAME
    )

    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Embedding dosyası bulunamadı:\n{embedding_path}"
        )

    embeddings = np.load(
        embedding_path
    )

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embedding matrisi iki boyutlu değil: "
            f"{embeddings.shape}"
        )

    if embeddings.shape[0] != article_count:
        raise ValueError(
            "Embedding satır sayısıyla makale sayısı uyuşmuyor.\n"
            f"Embedding satırı: {embeddings.shape[0]}\n"
            f"Makale sayısı   : {article_count}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embedding matrisinde NaN veya sonsuz değer var."
        )

    return embeddings.astype(
        np.float32,
        copy=False,
    )


def normalize_text(value: Any) -> str:
    """Metni tek satırlı ve temiz hâle getirir."""

    if not isinstance(value, str):
        return ""

    return " ".join(
        value.split()
    ).strip()


def markdown_escape(value: Any) -> str:
    """Markdown tablo işaretlerini temizler."""

    return (
        normalize_text(value)
        .replace("|", "\\|")
    )


def normalize_keywords(value: Any) -> List[str]:
    """Anahtar kelimeleri temiz bir listeye dönüştürür."""

    if not isinstance(value, list):
        return []

    keywords: List[str] = []

    for item in value:
        cleaned_item = normalize_text(item)

        if cleaned_item:
            keywords.append(cleaned_item)

    return keywords


def get_subject_items(
    article: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Makalenin subject kayıtlarını sözlük listesi olarak döndürür.

    Subjectler clustering işleminde kullanılmaz.
    Yalnızca sonuçların yorumlanmasında kullanılır.
    """

    raw_subjects = article.get("subjects")

    if not isinstance(raw_subjects, list):
        return []

    subject_items: List[Dict[str, Any]] = []

    for raw_subject in raw_subjects:
        if isinstance(raw_subject, dict):
            subject_items.append(raw_subject)
            continue

        if isinstance(raw_subject, str):
            try:
                parsed_subject = json.loads(
                    raw_subject
                )
            except json.JSONDecodeError:
                continue

            if isinstance(parsed_subject, dict):
                subject_items.append(
                    parsed_subject
                )

    return subject_items


def get_subject_names(
    article: Dict[str, Any],
) -> List[str]:
    """Subjectlerin okunabilir adlarını döndürür."""

    names: List[str] = []

    for subject in get_subject_items(article):
        full_name = subject.get("fullName")
        name = subject.get("name")

        if (
            isinstance(full_name, str)
            and full_name.strip()
        ):
            names.append(
                full_name.strip()
            )
        elif (
            isinstance(name, str)
            and name.strip()
        ):
            names.append(
                name.strip()
            )

    return names


def get_database_names(
    article: Dict[str, Any],
) -> List[str]:
    """SCIENCE/SOCIAL gibi database değerlerini döndürür."""

    databases = article.get("databases")

    if not isinstance(databases, list):
        return []

    return [
        normalize_text(database)
        for database in databases
        if normalize_text(database)
    ]


def count_cluster_keywords(
    cluster_indices: np.ndarray,
    articles: List[Dict[str, Any]],
) -> List[Tuple[str, int]]:
    """
    Cluster içindeki mevcut TR Dizin anahtar kelimelerini sayar.

    Aynı anahtar kelime tek makalede tekrar geçiyorsa
    yalnızca bir kez sayılır.
    """

    keyword_counter = Counter()
    display_names: Dict[str, str] = {}

    for article_index in cluster_indices:
        article = articles[
            int(article_index)
        ]

        article_keywords = normalize_keywords(
            article.get("keywords_tr")
        )

        unique_normalized_keywords = set()

        for keyword in article_keywords:
            normalized_keyword = (
                keyword
                .casefold()
                .strip(" .,:;")
            )

            if not normalized_keyword:
                continue

            unique_normalized_keywords.add(
                normalized_keyword
            )

            display_names.setdefault(
                normalized_keyword,
                keyword,
            )

        for normalized_keyword in (
            unique_normalized_keywords
        ):
            keyword_counter[
                normalized_keyword
            ] += 1

    return [
        (
            display_names[normalized_keyword],
            count,
        )
        for normalized_keyword, count
        in keyword_counter.most_common(
            TOP_KEYWORD_COUNT
        )
    ]


def count_cluster_subjects(
    cluster_indices: np.ndarray,
    articles: List[Dict[str, Any]],
) -> List[Tuple[str, int]]:
    """Cluster içindeki subject adlarını sayar."""

    subject_counter = Counter()

    for article_index in cluster_indices:
        article = articles[
            int(article_index)
        ]

        # Aynı subjecti tek makalede bir kez say.
        unique_subjects = set(
            get_subject_names(article)
        )

        for subject_name in unique_subjects:
            subject_counter[
                subject_name
            ] += 1

    return subject_counter.most_common(
        TOP_SUBJECT_COUNT
    )


def count_cluster_databases(
    cluster_indices: np.ndarray,
    articles: List[Dict[str, Any]],
) -> List[Tuple[str, int]]:
    """Cluster içindeki database dağılımını hesaplar."""

    database_counter = Counter()

    for article_index in cluster_indices:
        article = articles[
            int(article_index)
        ]

        for database_name in set(
            get_database_names(article)
        ):
            database_counter[
                database_name
            ] += 1

    return database_counter.most_common()


def get_representative_indices(
    cluster_indices: np.ndarray,
    embeddings: np.ndarray,
    cluster_center: np.ndarray,
) -> List[int]:
    """
    KMeans merkezine en yakın makaleleri bulur.

    KMeans Öklid uzaklığını kullandığı için temsilci
    seçiminde de merkeze Öklid uzaklığı kullanılır.
    """

    cluster_vectors = embeddings[
        cluster_indices
    ]

    distances = np.linalg.norm(
        cluster_vectors - cluster_center,
        axis=1,
    )

    nearest_positions = np.argsort(
        distances
    )[:REPRESENTATIVE_ARTICLE_COUNT]

    return [
        int(
            cluster_indices[
                int(position)
            ]
        )
        for position in nearest_positions
    ]


def get_boundary_indices(
    cluster_indices: np.ndarray,
    sample_silhouettes: np.ndarray,
) -> List[int]:
    """
    Silhouette değeri en düşük makaleleri bulur.

    Düşük veya negatif silhouette:
    - makalenin küme sınırında bulunabileceğini,
    - başka bir kümeye de yakın olabileceğini,
    - çok alanlı veya belirsiz olabileceğini

    düşündürebilir.

    Ancak silhouette bir konu olasılığı değildir.
    """

    cluster_silhouettes = (
        sample_silhouettes[
            cluster_indices
        ]
    )

    lowest_positions = np.argsort(
        cluster_silhouettes
    )[:BOUNDARY_ARTICLE_COUNT]

    return [
        int(
            cluster_indices[
                int(position)
            ]
        )
        for position in lowest_positions
    ]


def run_clustering(
    embeddings: np.ndarray,
) -> Tuple[
    KMeans,
    np.ndarray,
    np.ndarray,
]:
    """TR-MTEB embeddingleri üzerinde KMeans çalıştırır."""

    print("=" * 80)
    print("TR-MTEB KMEANS CLUSTERING")
    print("=" * 80)

    print(f"\nMakale sayısı : {embeddings.shape[0]}")
    print(f"Embedding boyutu: {embeddings.shape[1]}")
    print(f"Cluster sayısı: {CLUSTER_COUNT}")

    model = KMeans(
        n_clusters=CLUSTER_COUNT,
        random_state=RANDOM_SEED,
        n_init=N_INIT,
    )

    labels = model.fit_predict(
        embeddings
    )

    print("\nMakale bazlı cosine silhouette hesaplanıyor...")

    sample_silhouettes = silhouette_samples(
        embeddings,
        labels,
        metric="cosine",
    )

    print(
        "Ortalama cosine silhouette: "
        f"{sample_silhouettes.mean():.4f}"
    )

    return (
        model,
        labels,
        sample_silhouettes,
    )


def build_cluster_summaries(
    articles: List[Dict[str, Any]],
    embeddings: np.ndarray,
    model: KMeans,
    labels: np.ndarray,
    sample_silhouettes: np.ndarray,
) -> List[Dict[str, Any]]:
    """Her cluster için yorumlama bilgilerini oluşturur."""

    summaries: List[Dict[str, Any]] = []

    for cluster_id in range(
        CLUSTER_COUNT
    ):
        cluster_indices = np.where(
            labels == cluster_id
        )[0]

        representative_indices = (
            get_representative_indices(
                cluster_indices=cluster_indices,
                embeddings=embeddings,
                cluster_center=model.cluster_centers_[
                    cluster_id
                ],
            )
        )

        boundary_indices = (
            get_boundary_indices(
                cluster_indices=cluster_indices,
                sample_silhouettes=sample_silhouettes,
            )
        )

        years = [
            articles[int(index)].get(
                "publication_year"
            )
            for index in cluster_indices
            if articles[int(index)].get(
                "publication_year"
            ) is not None
        ]

        cluster_silhouettes = (
            sample_silhouettes[
                cluster_indices
            ]
        )

        summaries.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": int(
                    len(cluster_indices)
                ),
                "mean_silhouette": float(
                    cluster_silhouettes.mean()
                ),
                "minimum_silhouette": float(
                    cluster_silhouettes.min()
                ),
                "maximum_silhouette": float(
                    cluster_silhouettes.max()
                ),
                "minimum_year": (
                    min(years)
                    if years
                    else None
                ),
                "maximum_year": (
                    max(years)
                    if years
                    else None
                ),
                "top_keywords": (
                    count_cluster_keywords(
                        cluster_indices,
                        articles,
                    )
                ),
                "top_subjects": (
                    count_cluster_subjects(
                        cluster_indices,
                        articles,
                    )
                ),
                "database_distribution": (
                    count_cluster_databases(
                        cluster_indices,
                        articles,
                    )
                ),
                "representative_indices": (
                    representative_indices
                ),
                "boundary_indices": (
                    boundary_indices
                ),
            }
        )

    return summaries


def save_assignments_csv(
    articles: List[Dict[str, Any]],
    embeddings: np.ndarray,
    model: KMeans,
    labels: np.ndarray,
    sample_silhouettes: np.ndarray,
) -> Path:
    """Her makalenin cluster atamasını CSV olarak kaydeder."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day15_tr_mteb_k30_assignments.csv"
    )

    fieldnames = [
        "row_index",
        "article_id",
        "cluster_id",
        "distance_to_cluster_center",
        "silhouette",
        "publication_year",
        "databases",
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

        for row_index, article in enumerate(
            articles
        ):
            cluster_id = int(
                labels[row_index]
            )

            distance = np.linalg.norm(
                embeddings[row_index]
                - model.cluster_centers_[
                    cluster_id
                ]
            )

            writer.writerow(
                {
                    "row_index": row_index,
                    "article_id": article.get(
                        "article_id",
                        "",
                    ),
                    "cluster_id": cluster_id,
                    "distance_to_cluster_center": float(
                        distance
                    ),
                    "silhouette": float(
                        sample_silhouettes[
                            row_index
                        ]
                    ),
                    "publication_year": article.get(
                        "publication_year",
                        "",
                    ),
                    "databases": " | ".join(
                        get_database_names(
                            article
                        )
                    ),
                    "title_tr": article.get(
                        "title_tr",
                        "",
                    ),
                    "keywords_tr": " | ".join(
                        normalize_keywords(
                            article.get(
                                "keywords_tr"
                            )
                        )
                    ),
                    "subjects": " | ".join(
                        get_subject_names(
                            article
                        )
                    ),
                }
            )

    return output_path


def save_labeling_template_csv(
    summaries: List[Dict[str, Any]],
    articles: List[Dict[str, Any]],
) -> Path:
    """
    Clusterlara insan tarafından konu adı vermek için
    doldurulabilir CSV şablonu oluşturur.
    """

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day15_cluster_labeling_template.csv"
    )

    fieldnames = [
        "cluster_id",
        "cluster_size",
        "mean_silhouette",
        "top_keywords",
        "top_subjects",
        "representative_titles",
        "human_topic_label",
        "reviewer_notes",
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

        for summary in summaries:
            representative_titles = [
                normalize_text(
                    articles[index].get(
                        "title_tr"
                    )
                )
                for index in summary[
                    "representative_indices"
                ]
            ]

            writer.writerow(
                {
                    "cluster_id": summary[
                        "cluster_id"
                    ],
                    "cluster_size": summary[
                        "cluster_size"
                    ],
                    "mean_silhouette": summary[
                        "mean_silhouette"
                    ],
                    "top_keywords": " | ".join(
                        (
                            f"{keyword} ({count})"
                        )
                        for keyword, count
                        in summary[
                            "top_keywords"
                        ]
                    ),
                    "top_subjects": " | ".join(
                        (
                            f"{subject} ({count})"
                        )
                        for subject, count
                        in summary[
                            "top_subjects"
                        ]
                    ),
                    "representative_titles": (
                        " || ".join(
                            representative_titles
                        )
                    ),
                    "human_topic_label": "",
                    "reviewer_notes": "",
                }
            )

    return output_path


def save_summary_json(
    summaries: List[Dict[str, Any]],
) -> Path:
    """Cluster özetlerini JSON olarak kaydeder."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day15_tr_mteb_k30_summary.json"
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            summaries,
            json_file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


def save_markdown_report(
    summaries: List[Dict[str, Any]],
    articles: List[Dict[str, Any]],
    sample_silhouettes: np.ndarray,
) -> Path:
    """Cluster yorumlama raporunu Markdown olarak oluşturur."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day15_tr_mteb_k30_report.md"
    )

    lines: List[str] = [
        "# TR-MTEB — KMeans k=30 Cluster Yorumlama Raporu",
        "",
        (
            "Clustering yalnızca Türkçe abstractlardan üretilmiş "
            "TR-MTEB embeddingleri üzerinde yapılmıştır."
        ),
        "",
        (
            "Keywords ve subject alanları clustering sırasında "
            "kullanılmamıştır. Yalnızca clusterları sonradan "
            "yorumlamak için gösterilmektedir."
        ),
        "",
        (
            "Cluster numaralarının kendi başına konu anlamı yoktur. "
            "Konu isimleri temsilci makaleler incelendikten sonra "
            "insan tarafından verilmelidir."
        ),
        "",
    ]

    for summary in summaries:
        cluster_id = summary["cluster_id"]

        top_keywords = ", ".join(
            f"{keyword} ({count})"
            for keyword, count
            in summary["top_keywords"]
        ) or "Yeterli keyword yok"

        top_subjects = ", ".join(
            f"{subject} ({count})"
            for subject, count
            in summary["top_subjects"]
        ) or "Subject bilgisi yok"

        database_distribution = ", ".join(
            f"{database} ({count})"
            for database, count
            in summary["database_distribution"]
        ) or "Database bilgisi yok"

        lines.extend(
            [
                "---",
                "",
                f"## Cluster {cluster_id}",
                "",
                f"**Makale sayısı:** {summary['cluster_size']}",
                "",
                (
                    f"**Ortalama silhouette:** "
                    f"{summary['mean_silhouette']:.4f}"
                ),
                "",
                (
                    f"**Silhouette aralığı:** "
                    f"{summary['minimum_silhouette']:.4f} "
                    f"– {summary['maximum_silhouette']:.4f}"
                ),
                "",
                (
                    f"**Yıl aralığı:** "
                    f"{summary['minimum_year']}–"
                    f"{summary['maximum_year']}"
                ),
                "",
                f"**Database dağılımı:** {database_distribution}",
                "",
                f"**Öne çıkan keywords:** {top_keywords}",
                "",
                f"**Baskın subjectler:** {top_subjects}",
                "",
                "**İnsan tarafından verilecek konu adı:** _Henüz belirlenmedi_",
                "",
                "### Merkeze en yakın temsilci makaleler",
                "",
                "| Sıra | Makale | Keywords | Subjectler |",
                "|---:|---|---|---|",
            ]
        )

        for rank, article_index in enumerate(
            summary["representative_indices"],
            start=1,
        ):
            article = articles[
                article_index
            ]

            lines.append(
                f"| {rank} "
                f"| {markdown_escape(article.get('title_tr'))} "
                f"| {markdown_escape(', '.join(normalize_keywords(article.get('keywords_tr')))) or '-'} "
                f"| {markdown_escape(', '.join(get_subject_names(article))) or '-'} |"
            )

        lines.extend(
            [
                "",
                "### Küme sınırında olabilecek makaleler",
                "",
                (
                    "Bu makaleler cluster içindeki en düşük "
                    "silhouette değerlerine sahiptir."
                ),
                "",
                "| Silhouette | Makale | Keywords | Subjectler |",
                "|---:|---|---|---|",
            ]
        )

        for article_index in summary[
            "boundary_indices"
        ]:
            article = articles[
                article_index
            ]

            lines.append(
                f"| {sample_silhouettes[article_index]:.4f} "
                f"| {markdown_escape(article.get('title_tr'))} "
                f"| {markdown_escape(', '.join(normalize_keywords(article.get('keywords_tr')))) or '-'} "
                f"| {markdown_escape(', '.join(get_subject_names(article))) or '-'} |"
            )

        lines.append("")

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as markdown_file:
        markdown_file.write(
            "\n".join(lines)
        )

    return output_path


def create_cluster_size_chart(
    summaries: List[Dict[str, Any]],
) -> Path:
    """Cluster büyüklüklerini çubuk grafik olarak kaydeder."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day15_tr_mteb_k30_cluster_sizes.png"
    )

    cluster_ids = [
        summary["cluster_id"]
        for summary in summaries
    ]

    cluster_sizes = [
        summary["cluster_size"]
        for summary in summaries
    ]

    plt.figure(
        figsize=(14, 7)
    )

    plt.bar(
        cluster_ids,
        cluster_sizes,
    )

    plt.title(
        "TR-MTEB KMeans k=30 Cluster Büyüklükleri"
    )

    plt.xlabel(
        "Cluster ID"
    )

    plt.ylabel(
        "Makale sayısı"
    )

    plt.xticks(
        cluster_ids
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()

    return output_path


def print_terminal_summary(
    summaries: List[Dict[str, Any]],
) -> None:
    """Clusterların kısa özetini terminalde gösterir."""

    print("\n" + "=" * 80)
    print("CLUSTER ÖZETLERİ")
    print("=" * 80)

    for summary in summaries:
        top_keywords = ", ".join(
            keyword
            for keyword, _
            in summary["top_keywords"][:5]
        )

        top_subjects = ", ".join(
            subject
            for subject, _
            in summary["top_subjects"][:3]
        )

        print(
            f"\nCluster {summary['cluster_id']:2} "
            f"| n={summary['cluster_size']:3} "
            f"| silhouette="
            f"{summary['mean_silhouette']:.4f}"
        )

        print(
            f"  Keywords: "
            f"{top_keywords or '-'}"
        )

        print(
            f"  Subjects: "
            f"{top_subjects or '-'}"
        )


def main() -> None:
    articles = load_articles()

    embeddings = load_embeddings(
        article_count=len(articles)
    )

    (
        model,
        labels,
        sample_silhouettes,
    ) = run_clustering(
        embeddings=embeddings
    )

    summaries = build_cluster_summaries(
        articles=articles,
        embeddings=embeddings,
        model=model,
        labels=labels,
        sample_silhouettes=sample_silhouettes,
    )

    print_terminal_summary(
        summaries
    )

    assignments_path = save_assignments_csv(
        articles=articles,
        embeddings=embeddings,
        model=model,
        labels=labels,
        sample_silhouettes=sample_silhouettes,
    )

    labeling_template_path = (
        save_labeling_template_csv(
            summaries=summaries,
            articles=articles,
        )
    )

    summary_json_path = save_summary_json(
        summaries
    )

    markdown_path = save_markdown_report(
        summaries=summaries,
        articles=articles,
        sample_silhouettes=sample_silhouettes,
    )

    chart_path = create_cluster_size_chart(
        summaries
    )

    print("\n" + "=" * 80)
    print("DAY 15 TAMAMLANDI")
    print("=" * 80)

    print(
        f"\nMakale atamaları:\n"
        f"{assignments_path}"
    )

    print(
        f"\nKonu etiketleme şablonu:\n"
        f"{labeling_template_path}"
    )

    print(
        f"\nCluster özet JSON:\n"
        f"{summary_json_path}"
    )

    print(
        f"\nOkunabilir Markdown raporu:\n"
        f"{markdown_path}"
    )

    print(
        f"\nCluster büyüklük grafiği:\n"
        f"{chart_path}"
    )


if __name__ == "__main__":
    main()