#!/usr/bin/env python3
"""Bağımsız 5.000 TR Dizin makalesini toplar ve TR-MTEB embedding üretir."""

import argparse
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import requests
import torch
from requests.adapters import HTTPAdapter
from sentence_transformers import SentenceTransformer
from urllib3.util.retry import Retry


SEARCH_URL = "https://search.trdizin.gov.tr/api/defaultSearch/publication/"
MODEL_ID = "trmteb/turkish-embedding-model-fine-tuned"
TARGET_COUNT = 5000
MIN_ABSTRACT_CHARACTERS = 200
PAGE_LIMIT = 100
REQUEST_DELAY_SECONDS = 0.35
YEARS = list(range(2008, 2026))
QUERY_TERMS = [
    "sağlık", "eğitim", "bilim", "toplum", "ekonomi", "hukuk",
    "mühendislik", "çevre", "tarih", "sanat", "teknoloji", "araştırma",
]
PAGES_PER_QUERY = 3
TARGET_CANDIDATES = 6200
SELECTION_SEED = 2029


def root() -> Path:
    return Path(__file__).resolve().parents[3]


def paths() -> Dict[str, Path]:
    return {
        "pilot": root() / "data/processed/pilot_articles_1000.jsonl",
        "dataset": root() / "data/processed/validation_articles_5000.jsonl",
        "raw": root() / "data/raw/day29_validation_pages",
        "checkpoint": root() / "data/raw/day29_validation_candidates.jsonl",
        "progress": root() / "data/raw/day29_validation_progress.json",
        "embedding": root() / "research/outputs/day29_embeddings/tr_mteb_validation_5000.npy",
        "embedding_meta": root() / "research/outputs/day29_embeddings/tr_mteb_validation_5000.json",
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def atomic_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def atomic_numpy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def normalize_text(value: Any) -> str:
    return " ".join(value.split()).strip() if isinstance(value, str) else ""


def normalize_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [cleaned for cleaned in (normalize_text(item) for item in value) if cleaned]
    cleaned = normalize_text(value)
    return [cleaned] if cleaned else []


def normalize_subjects(value: Any) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    if not isinstance(value, list):
        return result
    for item in value:
        if isinstance(item, dict):
            result.append(item)
        elif isinstance(item, str):
            try:
                parsed = json.loads(item)
                if isinstance(parsed, dict):
                    result.append(parsed)
            except json.JSONDecodeError:
                pass
    return result


def extract_article(hit: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(hit, dict) or not isinstance(hit.get("_source"), dict):
        return None
    source = hit["_source"]
    abstracts = source.get("abstracts", [])
    if not isinstance(abstracts, list):
        return None
    turkish = next((item for item in abstracts if isinstance(item, dict)
                    and item.get("language") == "TUR" and normalize_text(item.get("abstract"))), None)
    if turkish is None:
        return None
    abstract = normalize_text(turkish.get("abstract"))
    article_id = str(source.get("id") or hit.get("_id") or "").strip()
    if not article_id or len(abstract) < MIN_ABSTRACT_CHARACTERS:
        return None
    language = str(source.get("language") or "").upper()
    doc_type = str(source.get("docType") or "").upper()
    if language and language != "TUR":
        return None
    if doc_type and doc_type != "PAPER":
        return None
    return {
        "article_id": article_id,
        "title_tr": normalize_text(turkish.get("title")),
        "abstract_tr": abstract,
        "keywords_tr": normalize_list(turkish.get("keywords")),
        "publication_year": source.get("publicationYear"),
        "publication_language": "TUR",
        "document_type": "PAPER",
        "databases": source.get("databases") if isinstance(source.get("databases"), list) else [],
        "subjects": normalize_subjects(source.get("subjects")),
        "abstract_character_count": len(abstract),
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("Bozuk JSONL: %s satır %d" % (path, line_number)) from error
            if isinstance(value, dict):
                rows.append(value)
    return rows


def pilot_ids() -> Set[str]:
    return set(str(row.get("article_id", "")).strip() for row in load_jsonl(paths()["pilot"]))


def session_with_retry() -> requests.Session:
    retry = Retry(total=5, connect=5, read=5, backoff_factor=1.0,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset(["GET"]), raise_on_status=False)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"Accept": "application/json", "User-Agent": "trdizin-semantic-lab/0.2"})
    return session


def request_tasks() -> List[Tuple[int, str, int]]:
    tasks: List[Tuple[int, str, int]] = []
    for offset in range(len(QUERY_TERMS)):
        for page in range(1, PAGES_PER_QUERY + 1):
            for year_position, year in enumerate(YEARS):
                query = QUERY_TERMS[(offset + year_position) % len(QUERY_TERMS)]
                tasks.append((year, query, page))
    return tasks


def raw_path(year: int, query: str, page: int) -> Path:
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]
    return paths()["raw"] / ("year_%d_query_%s_page_%02d.json" % (year, digest, page))


def fetch_or_load(session: requests.Session, year: int, query: str, page: int) -> Dict[str, Any]:
    target = raw_path(year, query, page)
    if target.exists():
        with target.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    params = {"q": query, "order": "publicationYear-DESC", "page": page, "limit": PAGE_LIMIT,
              "facet-documentType": "PAPER", "facet-publicationLanguage": "TUR",
              "facet-publication_year": year}
    response = session.get(SEARCH_URL, params=params, timeout=(15, 75))
    response.raise_for_status()
    data = response.json()
    atomic_json(target, data)
    time.sleep(REQUEST_DELAY_SECONDS)
    return data


def category(article: Dict[str, Any]) -> str:
    databases = set(str(value).upper() for value in article.get("databases", []))
    if "SCIENCE" in databases and "SOCIAL" in databases:
        return "BOTH"
    if "SCIENCE" in databases:
        return "SCIENCE"
    if "SOCIAL" in databases:
        return "SOCIAL"
    return "OTHER"


def balanced_select(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for article in candidates:
        key = (str(article.get("publication_year", "unknown")), category(article))
        buckets.setdefault(key, []).append(article)
    random_generator = random.Random(SELECTION_SEED)
    for values in buckets.values():
        values.sort(key=lambda row: row["article_id"])
        random_generator.shuffle(values)
    selected: List[Dict[str, Any]] = []
    selected_abstracts: Set[str] = set()
    keys = sorted(buckets)
    while len(selected) < TARGET_COUNT:
        changed = False
        for key in keys:
            if buckets[key]:
                article = buckets[key].pop()
                abstract_hash = hashlib.sha256(article["abstract_tr"].encode("utf-8")).hexdigest()
                if abstract_hash in selected_abstracts:
                    changed = True
                    continue
                selected.append(article)
                selected_abstracts.add(abstract_hash)
                changed = True
                if len(selected) == TARGET_COUNT:
                    break
        if not changed:
            break
    if len(selected) != TARGET_COUNT:
        raise RuntimeError("5.000 seçim için aday yetersiz: %d" % len(selected))
    return sorted(selected, key=lambda row: (str(row.get("publication_year", "")), row["article_id"]))


def validate_dataset(rows: List[Dict[str, Any]], excluded: Set[str]) -> None:
    ids = [str(row.get("article_id", "")) for row in rows]
    if len(rows) != TARGET_COUNT or len(set(ids)) != TARGET_COUNT:
        raise ValueError("Validation veri seti tam 5.000 benzersiz kayıt değil.")
    if set(ids) & excluded:
        raise ValueError("Pilot veriyle article_id çakışması var.")
    abstract_hashes = [hashlib.sha256(str(row.get("abstract_tr", "")).encode("utf-8")).hexdigest()
                       for row in rows]
    if len(set(abstract_hashes)) != TARGET_COUNT:
        raise ValueError("Duplicate abstract bulundu; duplicate embedding riski var.")
    for row in rows:
        if row.get("document_type") != "PAPER" or row.get("publication_language") != "TUR":
            raise ValueError("Facet koşulu ihlali: %s" % row.get("article_id"))
        if not normalize_text(row.get("abstract_tr")):
            raise ValueError("Türkçe abstract eksik: %s" % row.get("article_id"))


def build_dataset() -> List[Dict[str, Any]]:
    output = paths()["dataset"]
    excluded = pilot_ids()
    if output.exists():
        existing = load_jsonl(output)
        try:
            validate_dataset(existing, excluded)
            print("Geçerli 5.000 kayıtlık veri seti var; API aşaması atlandı.")
            return existing
        except ValueError as error:
            print("Mevcut veri seti checkpoint havuzundan yeniden seçilecek: %s" % error)
    checkpoint_rows = load_jsonl(paths()["checkpoint"])
    by_id: Dict[str, Dict[str, Any]] = {str(row["article_id"]): row for row in checkpoint_rows
                                      if str(row.get("article_id", "")) not in excluded}
    tasks = request_tasks()
    completed = 0
    api = session_with_retry()
    try:
        for task_number, (year, query, page) in enumerate(tasks, 1):
            if len(by_id) >= TARGET_CANDIDATES:
                break
            data = fetch_or_load(api, year, query, page)
            hits = data.get("hits", {}).get("hits", [])
            added = 0
            if isinstance(hits, list):
                for hit in hits:
                    article = extract_article(hit)
                    if article is None or article["article_id"] in excluded or article["article_id"] in by_id:
                        continue
                    by_id[article["article_id"]] = article
                    added += 1
            atomic_jsonl(paths()["checkpoint"], sorted(by_id.values(), key=lambda row: row["article_id"]))
            completed = task_number
            atomic_json(paths()["progress"], {"completed_task": task_number, "total_tasks": len(tasks),
                                               "year": year, "query": query, "page": page,
                                               "candidate_count": len(by_id), "last_added": added})
            print("[%d/%d] yıl=%d sorgu=%s sayfa=%d eklenen=%d toplam=%d" %
                  (task_number, len(tasks), year, query, page, added, len(by_id)), flush=True)
    finally:
        api.close()
    candidates = list(by_id.values())
    if len(candidates) < TARGET_COUNT:
        raise RuntimeError("API görevleri bitti; 5.000 aday yok: %d (son görev %d)" % (len(candidates), completed))
    selected = balanced_select(candidates)
    validate_dataset(selected, excluded)
    atomic_jsonl(output, selected)
    print("Validation veri seti yazıldı: %s" % output)
    return selected


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def duplicate_vector_count(values: np.ndarray) -> int:
    contiguous = np.ascontiguousarray(values)
    row_type = np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    packed = contiguous.view(row_type).reshape(-1)
    return int(len(packed) - len(np.unique(packed)))


def validate_embeddings(values: np.ndarray) -> None:
    if values.shape != (TARGET_COUNT, 768):
        raise ValueError("Embedding şekli (5000, 768) değil: %s" % (values.shape,))
    if not np.isfinite(values).all():
        raise ValueError("Embedding NaN/sonsuz içeriyor.")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Embeddingler normalize değil.")
    duplicates = duplicate_vector_count(values)
    if duplicates:
        raise ValueError("Duplicate embedding vektörü bulundu: %d" % duplicates)


def build_embeddings(articles: List[Dict[str, Any]]) -> np.ndarray:
    target = paths()["embedding"]
    current_hash = dataset_sha256(paths()["dataset"])
    if target.exists() and paths()["embedding_meta"].exists():
        values = np.load(str(target))
        with paths()["embedding_meta"].open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("dataset_sha256") == current_hash:
            validate_embeddings(values)
            print("Geçerli embedding mevcut; yeniden üretilmedi.")
            return values
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_sizes = [32, 16, 8, 4] if device == "cuda" else [16, 8, 4]
    print("TR-MTEB yükleniyor; device=%s" % device)
    cache_root = (Path.home() / ".cache/huggingface/hub/"
                  "models--trmteb--turkish-embedding-model-fine-tuned/snapshots")
    snapshots = sorted(path for path in cache_root.glob("*") if (path / "modules.json").exists())
    model_source = str(snapshots[-1]) if snapshots else MODEL_ID
    model = SentenceTransformer(model_source, device=device, trust_remote_code=False)
    model.max_seq_length = 512
    texts = [str(row["abstract_tr"]) for row in articles]
    last_error: Optional[Exception] = None
    started = time.perf_counter()
    for batch_size in batch_sizes:
        try:
            values = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=True)
            break
        except RuntimeError as error:
            last_error = error
            if "out of memory" not in str(error).lower():
                raise
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        raise last_error or RuntimeError("Embedding üretilemedi.")
    values = values.astype(np.float32, copy=False)
    validate_embeddings(values)
    atomic_numpy(target, values)
    atomic_json(paths()["embedding_meta"], {"model_id": MODEL_ID, "max_seq_length": 512,
                                            "normalize_embeddings": True, "shape": list(values.shape),
                                            "dataset_sha256": current_hash, "device": device,
                                            "batch_size": batch_size,
                                            "elapsed_seconds": time.perf_counter() - started,
                                            "duplicate_vector_count": 0})
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument("--embedding-only", action="store_true")
    args = parser.parse_args()
    if args.embedding_only:
        articles = load_jsonl(paths()["dataset"])
        validate_dataset(articles, pilot_ids())
    else:
        articles = build_dataset()
    if not args.dataset_only:
        build_embeddings(articles)


if __name__ == "__main__":
    main()
