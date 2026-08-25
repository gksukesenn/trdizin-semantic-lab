#!/usr/bin/env python3
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
from trdizin_topic_pipeline.config import load_config, resolve_path
from trdizin_topic_pipeline.utils.io import banner
from trdizin_topic_pipeline.reporting.report_builder import build_report, load_json
def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config", default="configs/final_50k.json"); args=parser.parse_args(); config=load_config(Path(args.config)); root=resolve_path(config,"output_root")
    quality=load_json(root/"reports/dataset_quality_summary.json"); topics=load_json(root/"clustering/final_topic_pipeline_summary.json"); target=root/"reports/FINAL_50000_TOPIC_DISCOVERY_REPORT.md"
    banner("NİHAİ 50.000 MAKALE RAPORU"); build_report(config,quality,topics,target); print("Rapor yazıldı               : %s" % target)
if __name__ == "__main__": main()
