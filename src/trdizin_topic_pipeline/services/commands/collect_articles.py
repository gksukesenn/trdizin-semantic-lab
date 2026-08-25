#!/usr/bin/env python3
"""TR Dizin API'den resume destekli nihai veri setini toplar."""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[4]
from trdizin_topic_pipeline.data.api_client import build_session, fetch_page, request_tasks
from trdizin_topic_pipeline.config import ensure_output_directories, load_config, resolve_path
from trdizin_topic_pipeline.data.dataset import dataset_identity, exclusion_ids, extract_article, validate_core
from trdizin_topic_pipeline.utils.io import abstract_sha256, append_jsonl, atomic_json, banner, read_jsonl


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final_50k.json"); parser.add_argument("--target", type=int)
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--reset-checkpoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--max-requests", type=int)
    return parser.parse_args()


def main() -> None:
    args = arguments(); config = load_config(Path(args.config)); ensure_output_directories(config)
    target = int(args.target or config["target_article_count"]); dataset_path = resolve_path(config, "dataset")
    checkpoint_path = resolve_path(config, "checkpoint"); pilot = read_jsonl(resolve_path(config, "pilot_dataset"))
    validation = read_jsonl(resolve_path(config, "validation_dataset")); excluded = exclusion_ids(pilot) | exclusion_ids(validation)
    banner("50.000 MAKALE VERİ TOPLAMA"); print("Hedef                     : %s" % format(target, ",").replace(",", "."))
    existing = read_jsonl(dataset_path)
    if existing and len(existing) == target:
        validate_core(existing, target, exclusion_ids(pilot), exclusion_ids(validation))
        print("Geçerli %s kayıt mevcut; veri yeniden çekilmedi." % format(target, ",").replace(",", ".")); return
    if existing and not args.resume:
        raise RuntimeError("Kısmi veri seti var (%d kayıt). Devam etmek için --resume kullanın." % len(existing))
    if args.reset_checkpoint:
        if existing: raise RuntimeError("Veri seti varken checkpoint sıfırlanamaz; kullanıcı verisi korunuyor.")
        if checkpoint_path.exists(): os.replace(str(checkpoint_path), str(checkpoint_path.with_suffix(".json.bak")))
    ids, hashes = dataset_identity(existing); ids.update(excluded)
    checkpoint: Dict[str, Any] = {"next_task_index": 0, "accepted_count": len(existing), "seen_article_id_count": len(ids),
                                  "seen_abstract_hash_count": len(hashes), "last_successful_request": None,
                                  "error_count": 0, "retry_count": 0, "started_at": now(), "updated_at": now()}
    if args.resume and checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as handle: checkpoint.update(json.load(handle))
    tasks = request_tasks(config["api"]["years"], config["api"]["queries"], int(config["api"]["pages_per_query"]))
    start_index = int(checkpoint.get("next_task_index", 0)); max_requests = args.max_requests
    if args.dry_run: max_requests = min(max_requests or 1, 1)
    api = build_session(int(config["api"]["retry_count"])); requests_done = 0; started = time.time()
    try:
        for index in range(start_index, len(tasks)):
            if len(existing) >= target or (max_requests is not None and requests_done >= max_requests): break
            year, query, page = tasks[index]
            try: data = fetch_page(api, config["api"], year, query, page)
            except Exception:
                checkpoint["error_count"] = int(checkpoint.get("error_count", 0)) + 1; checkpoint["updated_at"] = now()
                if not args.dry_run: atomic_json(checkpoint_path, checkpoint)
                raise
            hits = data.get("hits", {}).get("hits", []); accepted: List[Dict[str, Any]] = []; skipped = 0
            if not isinstance(hits, list): raise ValueError("API hits.hits listesi bulunamadı.")
            for hit in hits:
                article = extract_article(hit, int(config["api"].get("minimum_abstract_characters", 1)))
                if article is None: skipped += 1; continue
                article_id = article["article_id"]; digest = abstract_sha256(article["abstract_tr"])
                if article_id in ids or digest in hashes: skipped += 1; continue
                ids.add(article_id); hashes.add(digest); accepted.append(article)
                if len(existing) + len(accepted) >= target: break
            if not args.dry_run:
                append_jsonl(dataset_path, accepted); existing.extend(accepted)
            requests_done += 1
            checkpoint.update({"next_task_index": index + 1, "accepted_count": len(existing), "seen_article_id_count": len(ids),
                               "seen_abstract_hash_count": len(hashes), "last_successful_request": {"year": year, "query": query, "page": page},
                               "last_added": len(accepted), "last_skipped": skipped, "updated_at": now()})
            if not args.dry_run: atomic_json(checkpoint_path, checkpoint)
            print("[%d/%d] yıl=%d sorgu=%s sayfa=%d kabul=%d atlanan=%d toplam=%d geçen=%.1f sn" %
                  (index + 1, len(tasks), year, query, page, len(accepted), skipped, len(existing), time.time() - started))
    finally: api.close()
    if args.dry_run: print("Dry-run tamamlandı; final dataset/checkpoint değiştirilmedi."); return
    if len(existing) == target:
        rows = read_jsonl(dataset_path); rows.sort(key=lambda row: (str(row.get("publication_year", "")), str(row["article_id"])))
        # Deterministik final sıralama atomic yeniden yazılır.
        temporary = dataset_path.with_suffix(".jsonl.sorted.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows: handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(str(temporary), str(dataset_path))
        validate_core(rows, target, exclusion_ids(pilot), exclusion_ids(validation)); print("Tam veri seti doğrulandı: %s" % dataset_path)
    else: print("Toplama durdu: %d/%d kayıt; --resume ile devam edilebilir." % (len(existing), target))


if __name__ == "__main__": main()
