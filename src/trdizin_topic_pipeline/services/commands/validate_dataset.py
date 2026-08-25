#!/usr/bin/env python3
"""Nihai veri seti kapsamı, deduplication ve token uzunluklarını doğrular."""
import argparse, json, os, sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List
ROOT = Path(__file__).resolve().parents[4]; os.environ.setdefault("MPLCONFIGDIR", "/tmp/trdizin-matplotlib")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from trdizin_topic_pipeline.config import ensure_output_directories, load_config, resolve_path
from trdizin_topic_pipeline.data.dataset import exclusion_ids, validate_core
from trdizin_topic_pipeline.utils.io import atomic_csv, atomic_json, atomic_text, banner, file_sha256, read_jsonl
from trdizin_topic_pipeline.data.validation import metadata_counts, numeric_summary, token_lengths

def plot(values: List[int], path: Path, title: str, xlabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); plt.figure(figsize=(9, 5)); plt.hist(values, bins=60, color="#3568a8", edgecolor="white")
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Makale sayısı"); plt.tight_layout(); plt.savefig(str(path), dpi=160); plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/final_50k.json"); parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(); config = load_config(Path(args.config)); ensure_output_directories(config); root = resolve_path(config, "output_root")
    dataset = resolve_path(config, "validation_dataset") if args.smoke_test else resolve_path(config, "dataset"); rows = read_jsonl(dataset)
    if args.smoke_test: rows = rows[:100]; output = root.parent / "smoke_test"
    else: output = root
    if not rows: raise ValueError("Doğrulanacak veri seti boş: %s" % dataset)
    banner("VERİ SETİ KALİTE DOĞRULAMASI"); expected = len(rows) if args.smoke_test else int(config["target_article_count"])
    pilot_ids = exclusion_ids(read_jsonl(resolve_path(config, "pilot_dataset")))
    validation_ids = set() if args.smoke_test else exclusion_ids(read_jsonl(resolve_path(config, "validation_dataset")))
    core = validate_core(rows, expected, pilot_ids if not args.smoke_test else set(), validation_ids)
    summary: Dict[str, Any] = dict(core); summary.update(metadata_counts(rows)); summary["dataset_sha256"] = file_sha256(dataset)
    lengths = token_lengths(rows, str(config["embedding"]["model_id"])); summary["token_length"] = numeric_summary(lengths)
    summary["over_512_token_count"] = sum(value > 512 for value in lengths); summary["over_512_token_rate"] = summary["over_512_token_count"] / float(len(rows))
    char_lengths = [len(str(row["abstract_tr"])) for row in rows]; p95 = summary["abstract_character_length"]["p95"]
    summary["suspicious_very_long_abstract_count"] = sum(value > max(20000, p95 * 3) for value in char_lengths)
    reports, figures = output / "reports", output / "figures"; atomic_json(reports / "dataset_quality_summary.json", summary)
    years = Counter(str(row.get("publication_year") or "Bilinmiyor") for row in rows)
    year_rows = [{"publication_year": year, "article_count": count} for year, count in sorted(years.items())]
    atomic_csv(reports / "dataset_quality_by_year.csv", year_rows, ["publication_year", "article_count"])
    markdown = "# Veri Seti Kalite Raporu\n\n- Satır: %d\n- Benzersiz article_id: %d\n- Benzersiz abstract SHA-256: %d\n- Pilot çakışması: %d\n- Validation çakışması: %d\n- Boş abstract: %d\n- Boş başlık: %d\n- Subject bulunan oran: %.2f%%\n- Keyword bulunan oran: %.2f%%\n- 512 token üzeri: %d (%.2f%%)\n- Dataset SHA-256: `%s`\n\nMetadata yalnız veri toplama kalitesini raporlar; embedding girdisi yalnızca `abstract_tr` alanıdır.\n" % (len(rows), core["unique_article_id_count"], core["unique_abstract_sha256_count"], core["pilot_id_overlap_count"], core["validation_id_overlap_count"], core["empty_abstract_count"], core["empty_title_count"], summary["subject_present_rate"] * 100, summary["keyword_present_rate"] * 100, summary["over_512_token_count"], summary["over_512_token_rate"] * 100, summary["dataset_sha256"])
    atomic_text(reports / "dataset_quality_report.md", markdown); plot(char_lengths, figures / "abstract_length_distribution.png", "Abstract karakter uzunluğu", "Karakter")
    plot(lengths, figures / "token_length_distribution.png", "TR-MTEB tokenizer uzunluğu", "Token")
    print("Doğrulanan kayıt           : %d\nDataset SHA-256            : %s\nRapor dizini               : %s" % (len(rows), summary["dataset_sha256"], reports))

if __name__ == "__main__": main()
