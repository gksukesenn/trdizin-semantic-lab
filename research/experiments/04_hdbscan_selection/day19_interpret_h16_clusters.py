import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CONFIG_ID = "H16"

TOP_KEYWORD_COUNT = 12
TOP_SUBJECT_COUNT = 8
REPRESENTATIVE_COUNT = 5
SURPRISING_NOISE_COUNT = 20


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


def normalize_text(value: Any) -> str:
    """Metni temizleyip tek satıra dönüştürür."""

    if not isinstance(value, str):
        return ""

    return " ".join(value.split()).strip()


def markdown_escape(value: Any) -> str:
    """Markdown tablo işaretlerini güvenli hâle getirir."""

    return normalize_text(value).replace("|", "\\|")


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


def parse_subject_item(value: Any) -> Optional[Dict[str, Any]]:
    """Subject kaydını sözlüğe dönüştürür."""

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        cleaned_value = value.strip()

        if not cleaned_value:
            return None

        try:
            parsed_value = json.loads(cleaned_value)
        except json.JSONDecodeError:
            return None

        if isinstance(parsed_value, dict):
            return parsed_value

    return None


def get_subject_names(
    article: Dict[str, Any],
) -> List[str]:
    """Makaledeki subjectlerin okunabilir adlarını döndürür."""

    raw_subjects = article.get("subjects")

    if not isinstance(raw_subjects, list):
        return []

    subject_names: List[str] = []

    for raw_subject in raw_subjects:
        subject = parse_subject_item(raw_subject)

        if subject is None:
            continue

        full_name = subject.get("fullName")
        name = subject.get("name")

        if isinstance(full_name, str) and full_name.strip():
            subject_names.append(full_name.strip())
        elif isinstance(name, str) and name.strip():
            subject_names.append(name.strip())

    return subject_names


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
        for line_number, line in enumerate(input_file, start=1):
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

    if len(articles) != 1000:
        raise ValueError(
            "1.000 makale bekleniyordu, "
            f"bulunan: {len(articles)}"
        )

    return articles


def load_h16_assignments() -> List[Dict[str, Any]]:
    """Day 17 dosyasından H16 atamalarını okur."""

    input_path = (
        get_project_root()
        / "research" / "outputs"
        / "day17_hdbscan_all_assignments.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"HDBSCAN atama dosyası bulunamadı:\n{input_path}"
        )

    required_columns = {
        "row_index",
        "article_id",
        "kmeans_cluster_id",
        "kmeans_silhouette",
        f"{CONFIG_ID}_label",
        f"{CONFIG_ID}_probability",
        f"{CONFIG_ID}_outlier_score",
    }

    rows: List[Dict[str, Any]] = []

    with input_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        available_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - available_columns

        if missing_columns:
            raise ValueError(
                "Eksik sütunlar: "
                + ", ".join(sorted(missing_columns))
            )

        for row in reader:
            rows.append(
                {
                    "row_index": int(row["row_index"]),
                    "article_id": row["article_id"],
                    "kmeans_cluster_id": int(
                        row["kmeans_cluster_id"]
                    ),
                    "kmeans_silhouette": float(
                        row["kmeans_silhouette"]
                    ),
                    "hdbscan_label": int(
                        row[f"{CONFIG_ID}_label"]
                    ),
                    "hdbscan_probability": float(
                        row[f"{CONFIG_ID}_probability"]
                    ),
                    "hdbscan_outlier_score": float(
                        row[f"{CONFIG_ID}_outlier_score"]
                    ),
                }
            )

    rows.sort(key=lambda row: row["row_index"])

    if len(rows) != 1000:
        raise ValueError(
            "1.000 HDBSCAN ataması bekleniyordu, "
            f"bulunan: {len(rows)}"
        )

    return rows


def validate_alignment(
    articles: List[Dict[str, Any]],
    assignments: List[Dict[str, Any]],
) -> None:
    """Makale ve embedding satır sıralarının eşleştiğini doğrular."""

    for row_index, (article, assignment) in enumerate(
        zip(articles, assignments)
    ):
        article_id = str(article.get("article_id", ""))

        if assignment["row_index"] != row_index:
            raise ValueError(
                f"row_index uyuşmazlığı: {row_index}"
            )

        if article_id != assignment["article_id"]:
            raise ValueError(
                f"article_id uyuşmazlığı: {row_index}"
            )


def count_keywords(
    indices: List[int],
    articles: List[Dict[str, Any]],
) -> List[Tuple[str, int]]:
    """Cluster içindeki anahtar kelimelerin sıklığını hesaplar."""

    counter: Counter[str] = Counter()
    display_names: Dict[str, str] = {}

    for article_index in indices:
        keywords = normalize_keywords(
            articles[article_index].get("keywords_tr")
        )

        unique_keywords = set()

        for keyword in keywords:
            normalized_keyword = (
                keyword.casefold().strip(" .,:;")
            )

            if not normalized_keyword:
                continue

            unique_keywords.add(normalized_keyword)

            display_names.setdefault(
                normalized_keyword,
                keyword,
            )

        for normalized_keyword in unique_keywords:
            counter[normalized_keyword] += 1

    return [
        (
            display_names[normalized_keyword],
            count,
        )
        for normalized_keyword, count
        in counter.most_common(TOP_KEYWORD_COUNT)
    ]


def count_subjects(
    indices: List[int],
    articles: List[Dict[str, Any]],
) -> List[Tuple[str, int]]:
    """Cluster içindeki subject dağılımını hesaplar."""

    counter: Counter[str] = Counter()

    for article_index in indices:
        unique_subjects = set(
            get_subject_names(
                articles[article_index]
            )
        )

        for subject_name in unique_subjects:
            counter[subject_name] += 1

    return counter.most_common(TOP_SUBJECT_COUNT)


def select_representatives(
    indices: List[int],
    assignments: List[Dict[str, Any]],
) -> List[int]:
    """
    Üyelik olasılığı yüksek ve outlier skoru düşük
    makaleleri temsilci olarak seçer.
    """

    return sorted(
        indices,
        key=lambda article_index: (
            -assignments[article_index][
                "hdbscan_probability"
            ],
            assignments[article_index][
                "hdbscan_outlier_score"
            ],
            assignments[article_index][
                "article_id"
            ],
        ),
    )[:REPRESENTATIVE_COUNT]


def build_cluster_summaries(
    articles: List[Dict[str, Any]],
    assignments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """H16'nın 30 clusterı için özet üretir."""

    cluster_ids = sorted(
        {
            assignment["hdbscan_label"]
            for assignment in assignments
            if assignment["hdbscan_label"] >= 0
        }
    )

    summaries: List[Dict[str, Any]] = []

    for cluster_id in cluster_ids:
        indices = [
            row_index
            for row_index, assignment
            in enumerate(assignments)
            if assignment["hdbscan_label"] == cluster_id
        ]

        probabilities = [
            assignments[index]["hdbscan_probability"]
            for index in indices
        ]

        outlier_scores = [
            assignments[index]["hdbscan_outlier_score"]
            for index in indices
        ]

        summaries.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": len(indices),
                "mean_probability": float(
                    np.mean(probabilities)
                ),
                "median_probability": float(
                    np.median(probabilities)
                ),
                "minimum_probability": float(
                    np.min(probabilities)
                ),
                "mean_outlier_score": float(
                    np.mean(outlier_scores)
                ),
                "top_keywords": count_keywords(
                    indices,
                    articles,
                ),
                "top_subjects": count_subjects(
                    indices,
                    articles,
                ),
                "representative_indices": (
                    select_representatives(
                        indices,
                        assignments,
                    )
                ),
            }
        )

    return summaries


def build_noise_rows(
    articles: List[Dict[str, Any]],
    assignments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """H16 tarafından noise bırakılan makaleleri listeler."""

    noise_rows: List[Dict[str, Any]] = []

    for row_index, assignment in enumerate(assignments):
        if assignment["hdbscan_label"] != -1:
            continue

        article = articles[row_index]

        kmeans_silhouette = assignment[
            "kmeans_silhouette"
        ]

        if kmeans_silhouette < 0:
            noise_type = (
                "KMeans de belirsizdi"
            )
        else:
            noise_type = (
                "KMeans pozitifken HDBSCAN noise"
            )

        noise_rows.append(
            {
                "row_index": row_index,
                "article_id": assignment["article_id"],
                "noise_type": noise_type,
                "kmeans_cluster_id": assignment[
                    "kmeans_cluster_id"
                ],
                "kmeans_silhouette": kmeans_silhouette,
                "hdbscan_outlier_score": assignment[
                    "hdbscan_outlier_score"
                ],
                "publication_year": article.get(
                    "publication_year",
                    "",
                ),
                "title_tr": article.get(
                    "title_tr",
                    "",
                ),
                "keywords_tr": " | ".join(
                    normalize_keywords(
                        article.get("keywords_tr")
                    )
                ),
                "subjects": " | ".join(
                    get_subject_names(article)
                ),
            }
        )

    return noise_rows


def save_cluster_summary_csv(
    summaries: List[Dict[str, Any]],
    articles: List[Dict[str, Any]],
) -> Path:
    """Cluster özetlerini konu etiketleme şablonuna kaydeder."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day19_h16_cluster_summary.csv"
    )

    fieldnames = [
        "cluster_id",
        "cluster_size",
        "mean_probability",
        "median_probability",
        "minimum_probability",
        "mean_outlier_score",
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
                    articles[index].get("title_tr")
                )
                for index in summary[
                    "representative_indices"
                ]
            ]

            writer.writerow(
                {
                    "cluster_id": summary["cluster_id"],
                    "cluster_size": summary["cluster_size"],
                    "mean_probability": summary[
                        "mean_probability"
                    ],
                    "median_probability": summary[
                        "median_probability"
                    ],
                    "minimum_probability": summary[
                        "minimum_probability"
                    ],
                    "mean_outlier_score": summary[
                        "mean_outlier_score"
                    ],
                    "top_keywords": " | ".join(
                        f"{keyword} ({count})"
                        for keyword, count
                        in summary["top_keywords"]
                    ),
                    "top_subjects": " | ".join(
                        f"{subject} ({count})"
                        for subject, count
                        in summary["top_subjects"]
                    ),
                    "representative_titles": " || ".join(
                        representative_titles
                    ),
                    "human_topic_label": "",
                    "reviewer_notes": "",
                }
            )

    return output_path


def save_noise_csv(
    noise_rows: List[Dict[str, Any]],
) -> Path:
    """Bütün H16 noise makalelerini CSV olarak kaydeder."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day19_h16_noise_articles.csv"
    )

    sorted_rows = sorted(
        noise_rows,
        key=lambda row: (
            row["noise_type"],
            -row["kmeans_silhouette"],
        ),
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(sorted_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(sorted_rows)

    return output_path


def save_markdown_report(
    summaries: List[Dict[str, Any]],
    noise_rows: List[Dict[str, Any]],
    articles: List[Dict[str, Any]],
    assignments: List[Dict[str, Any]],
) -> Path:
    """H16 cluster yorumlama raporunu oluşturur."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day19_h16_cluster_report.md"
    )

    negative_noise_count = sum(
        row["noise_type"] == "KMeans de belirsizdi"
        for row in noise_rows
    )

    positive_noise_count = (
        len(noise_rows) - negative_noise_count
    )

    lines: List[str] = [
        "# H16 HDBSCAN Cluster Yorumlama Raporu",
        "",
        "HDBSCAN, TR-MTEB embeddinglerinin UMAP ile 10 boyuta "
        "indirgenmiş gösterimi üzerinde çalıştırılmıştır.",
        "",
        "Keywords ve subjectler clustering girdisi değildir. "
        "Yalnızca sonuçları yorumlamak amacıyla gösterilir.",
        "",
        "## Genel sonuç",
        "",
        f"- Cluster sayısı: {len(summaries)}",
        f"- Clusterlanan makale: {1000 - len(noise_rows)}",
        f"- Noise makale: {len(noise_rows)}",
        (
            f"- KMeans'te de negatif olan noise: "
            f"{negative_noise_count}"
        ),
        (
            f"- KMeans pozitifken HDBSCAN noise olan: "
            f"{positive_noise_count}"
        ),
        "",
    ]

    for summary in summaries:
        keywords_text = ", ".join(
            f"{keyword} ({count})"
            for keyword, count
            in summary["top_keywords"]
        ) or "Yeterli keyword yok"

        subjects_text = ", ".join(
            f"{subject} ({count})"
            for subject, count
            in summary["top_subjects"]
        ) or "Subject bilgisi yok"

        lines.extend(
            [
                "---",
                "",
                f"## H16 Cluster {summary['cluster_id']}",
                "",
                f"**Makale sayısı:** {summary['cluster_size']}",
                "",
                (
                    f"**Ortalama üyelik gücü:** "
                    f"{summary['mean_probability']:.4f}"
                ),
                "",
                (
                    f"**Medyan üyelik gücü:** "
                    f"{summary['median_probability']:.4f}"
                ),
                "",
                (
                    f"**Ortalama outlier skoru:** "
                    f"{summary['mean_outlier_score']:.4f}"
                ),
                "",
                f"**Öne çıkan keywords:** {keywords_text}",
                "",
                f"**Baskın subjectler:** {subjects_text}",
                "",
                "**İnsan konu etiketi:** _Henüz belirlenmedi_",
                "",
                "### Çekirdek temsilci makaleler",
                "",
                "| Üyelik | Outlier | Makale | Keywords | Subjectler |",
                "|---:|---:|---|---|---|",
            ]
        )

        for article_index in summary[
            "representative_indices"
        ]:
            article = articles[article_index]
            assignment = assignments[article_index]

            lines.append(
                f"| {assignment['hdbscan_probability']:.4f} "
                f"| {assignment['hdbscan_outlier_score']:.4f} "
                f"| {markdown_escape(article.get('title_tr'))} "
                f"| {markdown_escape(', '.join(normalize_keywords(article.get('keywords_tr')))) or '-'} "
                f"| {markdown_escape(', '.join(get_subject_names(article))) or '-'} |"
            )

        lines.append("")

    surprising_noise = sorted(
        [
            row
            for row in noise_rows
            if row["noise_type"]
            == "KMeans pozitifken HDBSCAN noise"
        ],
        key=lambda row: row["kmeans_silhouette"],
        reverse=True,
    )[:SURPRISING_NOISE_COUNT]

    lines.extend(
        [
            "---",
            "",
            "## KMeans güçlü görünürken HDBSCAN'in noise bıraktığı örnekler",
            "",
            (
                "Bu makalelerin KMeans silhouette değeri pozitiftir; "
                "ancak H16 yoğunluk yapısında kararlı bir kümeye "
                "bağlanamamışlardır."
            ),
            "",
            "| KMeans silhouette | KMeans cluster | Makale | Subjectler |",
            "|---:|---:|---|---|",
        ]
    )

    for row in surprising_noise:
        lines.append(
            f"| {row['kmeans_silhouette']:.4f} "
            f"| {row['kmeans_cluster_id']} "
            f"| {markdown_escape(row['title_tr'])} "
            f"| {markdown_escape(row['subjects']) or '-'} |"
        )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as markdown_file:
        markdown_file.write("\n".join(lines))

    return output_path


def create_cluster_size_chart(
    summaries: List[Dict[str, Any]],
) -> Path:
    """H16 cluster büyüklüklerini görselleştirir."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day19_h16_cluster_sizes.png"
    )

    cluster_ids = [
        summary["cluster_id"]
        for summary in summaries
    ]

    cluster_sizes = [
        summary["cluster_size"]
        for summary in summaries
    ]

    plt.figure(figsize=(14, 7))
    plt.bar(cluster_ids, cluster_sizes)

    plt.title(
        "H16 HDBSCAN Cluster Büyüklükleri"
    )
    plt.xlabel("HDBSCAN Cluster ID")
    plt.ylabel("Makale sayısı")
    plt.xticks(cluster_ids)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close()

    return output_path


def print_summary(
    summaries: List[Dict[str, Any]],
    noise_rows: List[Dict[str, Any]],
) -> None:
    """Terminalde kısa sonuç gösterir."""

    negative_noise_count = sum(
        row["noise_type"] == "KMeans de belirsizdi"
        for row in noise_rows
    )

    print("\n" + "=" * 85)
    print("H16 CLUSTER YORUMLAMA ÖZETİ")
    print("=" * 85)

    print(f"\nCluster sayısı       : {len(summaries)}")
    print(
        f"Clusterlanan makale  : "
        f"{1000 - len(noise_rows)}"
    )
    print(f"Noise makale         : {len(noise_rows)}")
    print(
        f"KMeans negatif noise : "
        f"{negative_noise_count}"
    )
    print(
        f"KMeans pozitif noise : "
        f"{len(noise_rows) - negative_noise_count}"
    )

    for summary in summaries:
        top_subject = (
            summary["top_subjects"][0][0]
            if summary["top_subjects"]
            else "-"
        )

        print(
            f"\nCluster {summary['cluster_id']:2} "
            f"| n={summary['cluster_size']:3} "
            f"| üyelik={summary['mean_probability']:.4f}"
        )
        print(f"  Baskın subject: {top_subject}")


def main() -> None:
    print("=" * 80)
    print("DAY 19 — H16 HDBSCAN CLUSTER YORUMLAMA")
    print("=" * 80)

    articles = load_articles()
    assignments = load_h16_assignments()

    validate_alignment(
        articles=articles,
        assignments=assignments,
    )

    summaries = build_cluster_summaries(
        articles=articles,
        assignments=assignments,
    )

    noise_rows = build_noise_rows(
        articles=articles,
        assignments=assignments,
    )

    print_summary(
        summaries=summaries,
        noise_rows=noise_rows,
    )

    summary_path = save_cluster_summary_csv(
        summaries=summaries,
        articles=articles,
    )

    noise_path = save_noise_csv(
        noise_rows=noise_rows,
    )

    markdown_path = save_markdown_report(
        summaries=summaries,
        noise_rows=noise_rows,
        articles=articles,
        assignments=assignments,
    )

    chart_path = create_cluster_size_chart(
        summaries=summaries
    )

    print("\n" + "=" * 80)
    print("DAY 19 TAMAMLANDI")
    print("=" * 80)

    print(f"\nCluster özet CSV:\n{summary_path}")
    print(f"\nNoise makaleler:\n{noise_path}")
    print(f"\nOkunabilir rapor:\n{markdown_path}")
    print(f"\nCluster büyüklük grafiği:\n{chart_path}")


if __name__ == "__main__":
    main()