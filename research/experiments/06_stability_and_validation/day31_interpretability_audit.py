import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


METHOD_NAME = "umap10_hdbscan_leaf"
TOP_TITLE_COUNT = 5


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def output_directory() -> Path:
    path = project_root() / "research" / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError("Dosya bulunamadı: %s" % path)

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def first_existing(
    row: Dict[str, str],
    candidates: List[str],
) -> Optional[str]:
    for candidate in candidates:
        if candidate in row:
            return candidate

    return None


def get_float(
    row: Dict[str, str],
    candidates: List[str],
    default: float = 0.0,
) -> float:
    key = first_existing(row, candidates)

    if key is None:
        return default

    value = str(row.get(key, "")).strip()

    if not value:
        return default

    try:
        return float(value)
    except ValueError:
        return default


def get_int(
    row: Dict[str, str],
    candidates: List[str],
    default: int = 0,
) -> int:
    return int(
        round(
            get_float(
                row=row,
                candidates=candidates,
                default=float(default),
            )
        )
    )


def parse_optional_bool(value: Any) -> Optional[bool]:
    normalized = str(value).strip().casefold()

    if normalized in {
        "true",
        "1",
        "yes",
        "evet",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
        "hayır",
        "hayir",
    }:
        return False

    return None


def choose_representative_seed(
    run_rows: List[Dict[str, str]],
) -> Tuple[int, Dict[str, float]]:
    """
    En yüksek sonucu seçip cherry-picking yapmaz.

    Cluster sayısı, Top-1 ve Top-2 değerleri bakımından
    beş koşunun ortalamasına en yakın seed'i seçer.
    """

    method_rows = [
        row
        for row in run_rows
        if row.get("method_name") == METHOD_NAME
    ]

    if len(method_rows) != 5:
        raise ValueError(
            "%s için 5 run bekleniyordu, bulunan: %d"
            % (
                METHOD_NAME,
                len(method_rows),
            )
        )

    prepared_rows: List[Dict[str, float]] = []

    for row in method_rows:
        prepared_rows.append(
            {
                "seed": float(
                    get_int(
                        row,
                        ["seed"],
                    )
                ),
                "cluster_count": get_float(
                    row,
                    [
                        "train_cluster_count",
                        "cluster_count",
                    ],
                ),
                "top1": get_float(
                    row,
                    [
                        "probe_top1_metadata_consistency",
                        "holdout_top1_metadata_consistency",
                        "top1_metadata_consistency",
                    ],
                ),
                "top2": get_float(
                    row,
                    [
                        "probe_top2_metadata_consistency",
                        "holdout_top2_metadata_consistency",
                        "top2_metadata_consistency",
                    ],
                ),
            }
        )

    averages = {
        "cluster_count": mean(
            row["cluster_count"]
            for row in prepared_rows
        ),
        "top1": mean(
            row["top1"]
            for row in prepared_rows
        ),
        "top2": mean(
            row["top2"]
            for row in prepared_rows
        ),
    }

    cluster_scale = max(
        max(
            row["cluster_count"]
            for row in prepared_rows
        )
        - min(
            row["cluster_count"]
            for row in prepared_rows
        ),
        1.0,
    )

    top1_scale = max(
        max(row["top1"] for row in prepared_rows)
        - min(row["top1"] for row in prepared_rows),
        0.000001,
    )

    top2_scale = max(
        max(row["top2"] for row in prepared_rows)
        - min(row["top2"] for row in prepared_rows),
        0.000001,
    )

    for row in prepared_rows:
        row["representativeness_distance"] = (
            abs(
                row["cluster_count"]
                - averages["cluster_count"]
            )
            / cluster_scale
            + abs(
                row["top1"]
                - averages["top1"]
            )
            / top1_scale
            + abs(
                row["top2"]
                - averages["top2"]
            )
            / top2_scale
        )

    selected = min(
        prepared_rows,
        key=lambda row: (
            row["representativeness_distance"],
            row["seed"],
        ),
    )

    return int(selected["seed"]), averages


def normalize_percentage(value: float) -> float:
    """
    CSV değeri 0–1 biçimindeyse yüzdeye çevirir.
    Zaten 0–100 ise değiştirmez.
    """

    if 0.0 <= value <= 1.0:
        return value * 100.0

    return value


def build_cluster_rows(
    predictions: List[Dict[str, str]],
    selected_seed: int,
) -> Tuple[List[Dict[str, Any]], int]:
    selected_rows = [
        row
        for row in predictions
        if (
            row.get("method_name") == METHOD_NAME
            and get_int(row, ["seed"]) == selected_seed
        )
    ]

    if len(selected_rows) != 1000:
        raise ValueError(
            "Seçilen seed için 1000 probe tahmini "
            "bekleniyordu, bulunan: %d"
            % len(selected_rows)
        )

    grouped: Dict[int, List[Dict[str, str]]] = defaultdict(list)

    for row in selected_rows:
        cluster_id = get_int(
            row,
            [
                "primary_cluster",
                "predicted_cluster",
            ],
            default=-999,
        )

        grouped[cluster_id].append(row)

    summary_rows: List[Dict[str, Any]] = []

    for cluster_id in sorted(grouped):
        rows = grouped[cluster_id]

        topic_counter = Counter(
            str(row.get("primary_topic", "")).strip()
            for row in rows
            if str(row.get("primary_topic", "")).strip()
        )

        primary_topic = (
            topic_counter.most_common(1)[0][0]
            if topic_counter
            else "Konu adı bulunamadı"
        )

        direct_count = 0
        fallback_count = 0

        similarities: List[float] = []
        top1_values: List[bool] = []
        top2_values: List[bool] = []

        for row in rows:
            method_text = str(
                row.get(
                    "direct_or_fallback",
                    row.get(
                        "assignment_method",
                        "",
                    ),
                )
            ).casefold()

            if "fallback" in method_text or "centroid" in method_text:
                fallback_count += 1
            else:
                direct_count += 1

            similarities.append(
                get_float(
                    row,
                    [
                        "primary_similarity",
                        "primary_centroid_similarity",
                    ],
                )
            )

            top1_value = parse_optional_bool(
                row.get(
                    "top1_matches_metadata",
                    row.get(
                        "primary_topic_matches_metadata",
                        "",
                    ),
                )
            )

            top2_value = parse_optional_bool(
                row.get(
                    "top2_matches_metadata",
                    row.get(
                        "top2_topics_match_metadata",
                        "",
                    ),
                )
            )

            if top1_value is not None:
                top1_values.append(top1_value)

            if top2_value is not None:
                top2_values.append(top2_value)

        ranked_titles = sorted(
            rows,
            key=lambda row: get_float(
                row,
                [
                    "primary_similarity",
                    "primary_centroid_similarity",
                ],
            ),
            reverse=True,
        )

        representative_titles = []

        for row in ranked_titles:
            title = str(
                row.get(
                    "title_tr",
                    "",
                )
            ).strip()

            if (
                title
                and title not in representative_titles
            ):
                representative_titles.append(title)

            if len(representative_titles) == TOP_TITLE_COUNT:
                break

        probe_count = len(rows)

        direct_rate = (
            direct_count / probe_count
            if probe_count
            else 0.0
        )

        top1_rate = (
            sum(top1_values) / len(top1_values)
            if top1_values
            else None
        )

        top2_rate = (
            sum(top2_values) / len(top2_values)
            if top2_values
            else None
        )

        warnings: List[str] = []

        if probe_count < 5:
            warnings.append(
                "probe örneği çok az"
            )

        if (
            top1_rate is not None
            and len(top1_values) >= 5
            and top1_rate < 0.30
        ):
            warnings.append(
                "düşük metadata tutarlılığı"
            )

        if not primary_topic or "cluster" in primary_topic.casefold():
            warnings.append(
                "konu adı zayıf"
            )

        summary_rows.append(
            {
                "seed": selected_seed,
                "cluster_id": cluster_id,
                "primary_topic": primary_topic,
                "probe_article_count": probe_count,
                "direct_count": direct_count,
                "fallback_count": fallback_count,
                "direct_rate": direct_rate,
                "mean_primary_similarity": (
                    mean(similarities)
                    if similarities
                    else 0.0
                ),
                "metadata_labeled_count": len(
                    top1_values
                ),
                "top1_metadata_consistency": (
                    top1_rate
                ),
                "top2_metadata_consistency": (
                    top2_rate
                ),
                "representative_title_1": (
                    representative_titles[0]
                    if len(representative_titles) > 0
                    else ""
                ),
                "representative_title_2": (
                    representative_titles[1]
                    if len(representative_titles) > 1
                    else ""
                ),
                "representative_title_3": (
                    representative_titles[2]
                    if len(representative_titles) > 2
                    else ""
                ),
                "representative_title_4": (
                    representative_titles[3]
                    if len(representative_titles) > 3
                    else ""
                ),
                "representative_title_5": (
                    representative_titles[4]
                    if len(representative_titles) > 4
                    else ""
                ),
                "warning": " | ".join(warnings),
            }
        )

    return summary_rows, len(selected_rows)


def save_csv(
    rows: List[Dict[str, Any]],
) -> Path:
    path = (
        output_directory()
        / "day31_cluster_interpretability_summary.csv"
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    return path


def save_chart(
    rows: List[Dict[str, Any]],
) -> Path:
    path = (
        output_directory()
        / "day31_cluster_probe_sizes.png"
    )

    sorted_rows = sorted(
        rows,
        key=lambda row: int(
            row["probe_article_count"]
        ),
        reverse=True,
    )

    cluster_labels = [
        str(row["cluster_id"])
        for row in sorted_rows
    ]

    cluster_sizes = [
        int(row["probe_article_count"])
        for row in sorted_rows
    ]

    plt.figure(
        figsize=(16, 7)
    )

    plt.bar(
        range(len(cluster_sizes)),
        cluster_sizes,
    )

    plt.title(
        "Temsilci Seed İçin HDBSCAN Leaf "
        "Probe Cluster Boyutları"
    )

    plt.xlabel(
        "Clusterlar — büyükten küçüğe"
    )

    plt.ylabel(
        "Probe makale sayısı"
    )

    if len(cluster_sizes) <= 110:
        plt.xticks(
            range(len(cluster_labels)),
            cluster_labels,
            rotation=90,
            fontsize=7,
        )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=170,
    )

    plt.close()

    return path


def save_markdown(
    rows: List[Dict[str, Any]],
    selected_seed: int,
    averages: Dict[str, float],
) -> Path:
    path = (
        output_directory()
        / "day31_cluster_interpretability_report.md"
    )

    warning_rows = [
        row
        for row in rows
        if row["warning"]
    ]

    direct_total = sum(
        int(row["direct_count"])
        for row in rows
    )

    fallback_total = sum(
        int(row["fallback_count"])
        for row in rows
    )

    lines: List[str] = [
        "# HDBSCAN Leaf Cluster Yorumlanabilirlik Denetimi",
        "",
        "## Temsilci koşu",
        "",
        (
            "- Yöntem: `umap10_hdbscan_leaf`"
        ),
        (
            "- Temsilci seed: `%d`" % selected_seed
        ),
        (
            "- Seçim yöntemi: En yüksek skoru seçmek yerine "
            "cluster sayısı, Top-1 ve Top-2 bakımından beş "
            "koşunun ortalamasına en yakın seed seçildi."
        ),
        (
            "- Beş koşu ortalama cluster sayısı: %.2f"
            % averages["cluster_count"]
        ),
        (
            "- Beş koşu ortalama Top-1: %.2f%%"
            % normalize_percentage(
                averages["top1"]
            )
        ),
        (
            "- Beş koşu ortalama Top-2: %.2f%%"
            % normalize_percentage(
                averages["top2"]
            )
        ),
        "",
        "## Genel görünüm",
        "",
        "- İncelenen probe makalesi: 1000",
        "- Probe üzerinde görülen cluster: %d" % len(rows),
        "- Doğrudan HDBSCAN ataması: %d" % direct_total,
        "- Centroid fallback: %d" % fallback_total,
        "- Uyarı işaretlenen cluster: %d" % len(warning_rows),
        "",
        "## Cluster özetleri",
        "",
    ]

    for row in sorted(
        rows,
        key=lambda item: int(
            item["probe_article_count"]
        ),
        reverse=True,
    ):
        top1_value = row[
            "top1_metadata_consistency"
        ]

        top2_value = row[
            "top2_metadata_consistency"
        ]

        lines.extend(
            [
                (
                    "### Cluster %s — %s"
                    % (
                        row["cluster_id"],
                        row["primary_topic"],
                    )
                ),
                "",
                (
                    "- Probe makale sayısı: %s"
                    % row["probe_article_count"]
                ),
                (
                    "- Doğrudan / fallback: %s / %s"
                    % (
                        row["direct_count"],
                        row["fallback_count"],
                    )
                ),
                (
                    "- Ortalama centroid benzerliği: %.4f"
                    % float(
                        row[
                            "mean_primary_similarity"
                        ]
                    )
                ),
                (
                    "- Metadata Top-1: %s"
                    % (
                        "ölçülemedi"
                        if top1_value is None
                        else "%%%0.2f"
                        % (
                            float(top1_value)
                            * 100
                        )
                    )
                ),
                (
                    "- Metadata Top-2: %s"
                    % (
                        "ölçülemedi"
                        if top2_value is None
                        else "%%%0.2f"
                        % (
                            float(top2_value)
                            * 100
                        )
                    )
                ),
            ]
        )

        if row["warning"]:
            lines.append(
                "- **Uyarı:** %s"
                % row["warning"]
            )

        lines.append(
            "- Temsilci başlıklar:"
        )

        for title_index in range(
            1,
            TOP_TITLE_COUNT + 1,
        ):
            title = str(
                row.get(
                    "representative_title_%d"
                    % title_index,
                    "",
                )
            ).strip()

            if title:
                lines.append(
                    "  - %s" % title
                )

        lines.append("")

    lines.extend(
        [
            "## Yorumlama sınırları",
            "",
            (
                "- Bu rapor yalnızca sabit 1.000 probe "
                "makalesindeki görünümü özetler."
            ),
            (
                "- Cluster konu adları eğitim makalelerinin "
                "TR Dizin subject metadata alanlarından "
                "türetilen geçici adlardır."
            ),
            (
                "- Metadata tutarlılığı bağımsız bir "
                "sınıflandırma doğruluk oranı değildir."
            ),
            (
                "- Küçük clusterlar ve düşük tutarlılıklı "
                "clusterlar 50.000 makalelik aşamada yeniden "
                "incelenmelidir."
            ),
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return path


def save_json_summary(
    rows: List[Dict[str, Any]],
    selected_seed: int,
    averages: Dict[str, float],
) -> Path:
    path = (
        output_directory()
        / "day31_interpretability_summary.json"
    )

    summary = {
        "method_name": METHOD_NAME,
        "representative_seed": selected_seed,
        "selection_rule": (
            "Beş koşunun cluster count, Top-1 ve Top-2 "
            "ortalamalarına en yakın seed."
        ),
        "five_run_averages": averages,
        "probe_cluster_count": len(rows),
        "warning_cluster_count": sum(
            bool(row["warning"])
            for row in rows
        ),
        "clusters": rows,
    }

    path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return path


def main() -> None:
    runs_path = (
        output_directory()
        / "day30_finalist_runs.csv"
    )

    predictions_path = (
        output_directory()
        / "day30_probe_predictions.csv"
    )

    run_rows = read_csv(
        runs_path
    )

    prediction_rows = read_csv(
        predictions_path
    )

    (
        selected_seed,
        averages,
    ) = choose_representative_seed(
        run_rows
    )

    (
        cluster_rows,
        prediction_count,
    ) = build_cluster_rows(
        predictions=prediction_rows,
        selected_seed=selected_seed,
    )

    csv_path = save_csv(
        cluster_rows
    )

    chart_path = save_chart(
        cluster_rows
    )

    report_path = save_markdown(
        rows=cluster_rows,
        selected_seed=selected_seed,
        averages=averages,
    )

    json_path = save_json_summary(
        rows=cluster_rows,
        selected_seed=selected_seed,
        averages=averages,
    )

    print("=" * 85)
    print("DAY 31 — CLUSTER YORUMLANABİLİRLİK DENETİMİ")
    print("=" * 85)

    print(
        "\nTemsilci seed            : %d"
        % selected_seed
    )

    print(
        "İncelenen probe makalesi : %d"
        % prediction_count
    )

    print(
        "Probe cluster sayısı     : %d"
        % len(cluster_rows)
    )

    print(
        "Uyarılı cluster sayısı   : %d"
        % sum(
            bool(row["warning"])
            for row in cluster_rows
        )
    )

    print("\nDosyalar:")

    print(
        "- %s" % csv_path
    )

    print(
        "- %s" % chart_path
    )

    print(
        "- %s" % report_path
    )

    print(
        "- %s" % json_path
    )


if __name__ == "__main__":
    main()