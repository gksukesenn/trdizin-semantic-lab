"""Stable CSV and Markdown writers for retrieval benchmark outputs."""
import csv
from pathlib import Path
from typing import Any, Dict, List

def write_results_csv(
    rows: List[Dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "query",
                "rank",
                "qdrant_id",
                "article_id",
                "semantic_score",
                "metadata_relevant",
                "title_tr",
                "publication_year",
                "databases",
                "assignment_method",
                "primary_cluster",
                "primary_topic",
                "secondary_cluster",
                "secondary_topic",
                "similarity_margin",
                "human_judgment",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(
    rows: List[Dict[str, Any]],
    path: Path,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "query",
                "result_count",
                "precision_at_5",
                "precision_at_10",
                "mrr_at_10",
                "ndcg_at_10",
                "qdrant_seconds",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    query_summaries: List[Dict[str, Any]],
    path: Path,
) -> None:
    lines = [
        "# 50.000 Makale Semantic Search Benchmarkı",
        "",
        "Bu rapor TR Dizin subject metadata alanlarını keşifsel "
        "relevance göstergesi olarak kullanır.",
        "",
        "Bu değerler bağımsız insan değerlendirmesi veya gerçek "
        "retrieval accuracy değildir.",
        "",
        "## Genel sonuç",
        "",
        "- Sorgu sayısı: %d" % len(query_summaries),
        "- Ortalama Precision@5: %.4f"
        % mean(
            float(row["precision_at_5"])
            for row in query_summaries
        ),
        "- Ortalama Precision@10: %.4f"
        % mean(
            float(row["precision_at_10"])
            for row in query_summaries
        ),
        "- Ortalama MRR@10: %.4f"
        % mean(
            float(row["mrr_at_10"])
            for row in query_summaries
        ),
        "- Ortalama nDCG@10: %.4f"
        % mean(
            float(row["ndcg_at_10"])
            for row in query_summaries
        ),
        "",
        "## Sorgu sonuçları",
        "",
        "| ID | Sorgu | P@5 | P@10 | MRR@10 | nDCG@10 |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for row in query_summaries:
        lines.append(
            "| %s | %s | %.4f | %.4f | %.4f | %.4f |"
            % (
                row["query_id"],
                row["query"],
                float(row["precision_at_5"]),
                float(row["precision_at_10"]),
                float(row["mrr_at_10"]),
                float(row["ndcg_at_10"]),
            )
        )

    lines.extend(
        [
            "",
            "## Yorumlama sınırları",
            "",
            "- Subject metadata ground truth değildir.",
            "- Bir makalenin birden fazla subject alanı olabilir.",
            "- Primary ve secondary topic adları subject metadata "
            "üzerinden türetilen geçici konu adlarıdır.",
            "- `human_judgment` sütunu ileride sınırlı insan "
            "değerlendirmesi için boş bırakılmıştır.",
        ]
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


