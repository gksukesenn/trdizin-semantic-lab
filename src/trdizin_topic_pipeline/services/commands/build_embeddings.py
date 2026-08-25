#!/usr/bin/env python3
"""Nihai TR-MTEB embeddinglerini üretir."""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
from trdizin_topic_pipeline.config import ensure_output_directories, load_config, resolve_path
from trdizin_topic_pipeline.topics.embeddings import build_embeddings
from trdizin_topic_pipeline.utils.io import banner, read_jsonl

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/final_50k.json")
    parser.add_argument("--allow-cpu", action="store_true"); parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(); config = load_config(Path(args.config)); ensure_output_directories(config)
    dataset = resolve_path(config, "validation_dataset") if args.smoke_test else resolve_path(config, "dataset")
    rows = read_jsonl(dataset); expected = int(config["target_article_count"])
    if not args.smoke_test and len(rows) != expected: raise ValueError("Embedding için tam %d kayıt gerekir; bulunan %d." % (expected, len(rows)))
    banner("TR-MTEB EMBEDDING"); output = resolve_path(config, "embedding"); metadata = resolve_path(config, "embedding_metadata"); progress = resolve_path(config, "embedding_progress")
    if args.smoke_test:
        smoke = resolve_path(config, "output_root").parent / "smoke_test" / "embeddings"; smoke.mkdir(parents=True, exist_ok=True)
        output, metadata, progress = smoke / "smoke.npy", smoke / "smoke_metadata.json", smoke / "smoke_progress.json"
    build_embeddings(rows, dataset, output, metadata, progress, config["embedding"], args.allow_cpu, limit=32 if args.smoke_test else 0)

if __name__ == "__main__": main()
