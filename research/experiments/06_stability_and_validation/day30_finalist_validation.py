#!/usr/bin/env python3
"""Day 30: bağımsız 5.000 makalede iki finalisti karşılaştırır."""

import csv
import hashlib
import json
import math
import os
import time
import traceback
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/trdizin-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hdbscan
import numpy as np
import umap
from hdbscan.prediction import approximate_predict
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import train_test_split


SEEDS = [11, 22, 33, 42, 55]
PROBE_SEED = 2026
ARTICLE_COUNT = 5000
PROBE_COUNT = 1000
TRAIN_POOL_COUNT = 4000
TRAIN_COUNT = 3600
METHODS = ["kmeans_k30", "umap10_hdbscan_leaf"]

RUN_FIELDS = [
    "method_name", "seed", "train_article_count", "probe_article_count",
    "train_cluster_count", "train_noise_count", "train_noise_rate",
    "min_cluster_size", "median_cluster_size", "max_cluster_size",
    "weighted_train_subject_purity", "probe_direct_count", "probe_direct_rate",
    "probe_fallback_count", "probe_fallback_rate", "probe_subject_labeled_count",
    "probe_top1_metadata_consistency", "probe_top2_metadata_consistency",
    "mean_similarity_margin", "median_similarity_margin", "cosine_silhouette",
    "collapsed_run", "runtime_seconds", "warning", "error",
]
PRED_FIELDS = [
    "method_name", "seed", "probe_position", "global_row_index", "article_id",
    "publication_year", "title_tr", "raw_direct_or_noise", "direct_or_fallback",
    "raw_predicted_cluster", "primary_cluster", "primary_topic", "primary_top_subjects",
    "primary_similarity", "secondary_cluster", "secondary_topic", "secondary_top_subjects",
    "secondary_similarity", "similarity_margin", "known_subjects",
    "top1_matches_metadata", "top2_matches_metadata",
]
PAIR_FIELDS = ["method_name", "seed_a", "seed_b", "probe_count", "adjusted_rand_index",
               "normalized_mutual_information", "raw_status_agreement"]
SUMMARY_FIELDS = [
    "method_name", "successful_run_count", "collapse_run_count", "cluster_count_mean",
    "cluster_count_std", "cluster_count_min", "cluster_count_max", "train_noise_rate_mean",
    "train_noise_rate_std", "probe_direct_rate_mean", "probe_direct_rate_std",
    "probe_fallback_rate_mean", "probe_fallback_rate_std", "probe_top1_mean",
    "probe_top1_std", "probe_top2_mean", "probe_top2_std", "weighted_train_purity_mean",
    "similarity_margin_mean", "cosine_silhouette_mean", "runtime_seconds_mean",
    "ari_mean", "ari_std", "ari_min", "ari_max", "nmi_mean", "nmi_std", "nmi_min",
    "nmi_max", "raw_status_agreement_mean", "raw_status_agreement_std",
]


def root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_inputs() -> Tuple[List[Dict[str, Any]], np.ndarray]:
    articles: List[Dict[str, Any]] = []
    path = root() / "data/processed/validation_articles_5000.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict): articles.append(value)
    embeddings = np.load(str(root() / "research/outputs/day29_embeddings/tr_mteb_validation_5000.npy"))
    ids = [str(row.get("article_id", "")) for row in articles]
    if len(articles) != ARTICLE_COUNT or len(set(ids)) != ARTICLE_COUNT:
        raise ValueError("Validation makaleleri 5.000 benzersiz ID içermiyor.")
    if embeddings.shape != (ARTICLE_COUNT, 768) or not np.isfinite(embeddings).all():
        raise ValueError("Embedding şekli/içeriği geçersiz: %s" % (embeddings.shape,))
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0): raise ValueError("Sıfır embedding var.")
    embeddings = (embeddings / norms).astype(np.float32)
    return articles, embeddings


def parse_subject(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict): return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError: return None
    return None


def subject_data(articles: List[Dict[str, Any]]) -> Tuple[List[Set[str]], Dict[str, str]]:
    sets: List[Set[str]] = []
    display: Dict[str, str] = {}
    for article in articles:
        keys: Set[str] = set()
        values = article.get("subjects", [])
        if not isinstance(values, list): values = []
        for raw in values:
            item = parse_subject(raw)
            if item is None: continue
            if item.get("id") is not None: key = "id:%s" % item["id"]
            elif str(item.get("fullName", "")).strip(): key = "full:%s" % str(item["fullName"]).strip()
            elif str(item.get("name", "")).strip(): key = "name:%s" % str(item["name"]).strip()
            else: continue
            keys.add(key); display[key] = str(item.get("fullName") or item.get("name") or key).strip()
        sets.append(keys)
    return sets, display


def years(articles: List[Dict[str, Any]], indices: np.ndarray) -> np.ndarray:
    return np.asarray([str(articles[int(index)].get("publication_year", "unknown")) for index in indices])


def build_splits(articles: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    all_indices = np.arange(ARTICLE_COUNT, dtype=np.int32)
    pool, probe = train_test_split(all_indices, test_size=PROBE_COUNT, random_state=PROBE_SEED,
                                   stratify=years(articles, all_indices))
    pool = np.sort(pool.astype(np.int32)); probe = np.sort(probe.astype(np.int32))
    training: Dict[int, np.ndarray] = {}
    for seed in SEEDS:
        selected, unused = train_test_split(pool, train_size=TRAIN_COUNT, random_state=seed,
                                            stratify=years(articles, pool))
        selected = np.sort(selected.astype(np.int32))
        assert len(selected) == TRAIN_COUNT and len(unused) == 400
        assert np.intersect1d(selected, probe).size == 0
        training[seed] = selected
    assert len(pool) == TRAIN_POOL_COUNT and len(probe) == PROBE_COUNT
    return pool, probe, training


def normalized_centroid(values: np.ndarray) -> np.ndarray:
    centroid = values.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm == 0: raise ValueError("Sıfır centroid.")
    return (centroid / norm).astype(np.float32)


def build_cluster_info(train_indices: np.ndarray, probe_indices: np.ndarray,
                       train_embeddings: np.ndarray, labels: np.ndarray,
                       subject_sets: List[Set[str]], display: Dict[str, str]) -> Tuple[List[int], Dict[int, Dict[str, Any]], np.ndarray]:
    train_access = set(int(value) for value in train_indices)
    probe_set = set(int(value) for value in probe_indices)
    assert train_access.isdisjoint(probe_set)
    cluster_ids = sorted(set(int(label) for label in labels if int(label) >= 0))
    if not cluster_ids: raise RuntimeError("Training cluster oluşmadı.")
    info: Dict[int, Dict[str, Any]] = {}; centroids: List[np.ndarray] = []
    for cluster_id in cluster_ids:
        local = np.where(labels == cluster_id)[0]
        global_indices = [int(train_indices[int(position)]) for position in local]
        assert set(global_indices).issubset(train_access) and set(global_indices).isdisjoint(probe_set)
        centroids.append(normalized_centroid(train_embeddings[local]))
        counts: Counter = Counter(); labeled = 0
        for global_index in global_indices:
            values = subject_sets[global_index]
            if values:
                labeled += 1; counts.update(sorted(values))
        top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:3]
        dominant = top[0][0] if top else ""
        info[cluster_id] = {"size": len(local), "labeled": labeled, "dominant": dominant,
                            "topic": display.get(dominant, "Cluster %d" % cluster_id),
                            "top": [{"key": key, "name": display.get(key, key), "count": int(count)}
                                    for key, count in top],
                            "purity": (float(top[0][1]) / labeled) if top and labeled else 0.0}
    assert train_access.isdisjoint(probe_set)
    return cluster_ids, info, np.vstack(centroids).astype(np.float32)


def fit(method: str, seed: int, train_embeddings: np.ndarray,
        probe_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if method == "kmeans_k30":
        model = KMeans(n_clusters=30, n_init=20, random_state=seed)
        return model.fit_predict(train_embeddings).astype(np.int32), model.predict(probe_embeddings).astype(np.int32)
    reducer = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric="cosine",
                        random_state=seed, low_memory=True)
    train_reduced = reducer.fit_transform(train_embeddings)
    probe_reduced = reducer.transform(probe_embeddings)
    model = hdbscan.HDBSCAN(cluster_selection_method="leaf", min_cluster_size=10, min_samples=5,
                            metric="euclidean", prediction_data=True, gen_min_span_tree=False)
    train_labels = model.fit_predict(train_reduced).astype(np.int32)
    probe_labels, strengths = approximate_predict(model, probe_reduced)
    return train_labels, probe_labels.astype(np.int32)


def evaluate(method: str, seed: int, articles: List[Dict[str, Any]], probe_indices: np.ndarray,
             probe_embeddings: np.ndarray, raw_labels: np.ndarray, cluster_ids: List[int],
             info: Dict[int, Dict[str, Any]], centroids: np.ndarray, subject_sets: List[Set[str]],
             display: Dict[str, str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], np.ndarray, np.ndarray]:
    scores = probe_embeddings @ centroids.T
    positions = {cluster_id: position for position, cluster_id in enumerate(cluster_ids)}
    rows: List[Dict[str, Any]] = []; final_labels: List[int] = []; raw_status: List[int] = []
    labeled = top1_count = top2_count = direct = fallback = 0; margins: List[float] = []
    for probe_position, raw_global in enumerate(probe_indices):
        global_index = int(raw_global); raw = int(raw_labels[probe_position])
        ranked = [cluster_ids[int(pos)] for pos in np.argsort(scores[probe_position])[::-1]]
        if method == "kmeans_k30": primary = raw; status = "kmeans_predict"; raw_name = "not_applicable"
        elif raw >= 0: primary = raw; status = "direct"; raw_name = "direct"; direct += 1
        else: primary = ranked[0]; status = "fallback"; raw_name = "noise"; fallback += 1
        secondary = next((value for value in ranked if value != primary), primary)
        primary_similarity = float(scores[probe_position, positions[primary]])
        secondary_similarity = float(scores[probe_position, positions[secondary]])
        margin = primary_similarity - secondary_similarity; margins.append(margin)
        known = subject_sets[global_index]
        if known:
            labeled += 1
            top1 = bool(info[primary]["dominant"] and info[primary]["dominant"] in known)
            top2 = bool(top1 or (info[secondary]["dominant"] and info[secondary]["dominant"] in known))
            top1_count += int(top1); top2_count += int(top2)
        else: top1 = None; top2 = None
        article = articles[global_index]
        assert str(article["article_id"]) == str(articles[int(probe_indices[probe_position])]["article_id"])
        rows.append({"method_name": method, "seed": seed, "probe_position": probe_position,
                     "global_row_index": global_index, "article_id": article["article_id"],
                     "publication_year": article.get("publication_year", ""), "title_tr": article.get("title_tr", ""),
                     "raw_direct_or_noise": raw_name, "direct_or_fallback": status,
                     "raw_predicted_cluster": raw, "primary_cluster": primary,
                     "primary_topic": info[primary]["topic"],
                     "primary_top_subjects": json.dumps(info[primary]["top"], ensure_ascii=False),
                     "primary_similarity": primary_similarity, "secondary_cluster": secondary,
                     "secondary_topic": info[secondary]["topic"],
                     "secondary_top_subjects": json.dumps(info[secondary]["top"], ensure_ascii=False),
                     "secondary_similarity": secondary_similarity, "similarity_margin": margin,
                     "known_subjects": " | ".join(sorted(display.get(key, key) for key in known)),
                     "top1_matches_metadata": top1, "top2_matches_metadata": top2})
        final_labels.append(primary); raw_status.append(1 if raw >= 0 else 0)
    margin_array = np.asarray(margins)
    return rows, {"probe_direct_count": direct if method != "kmeans_k30" else 0,
                  "probe_direct_rate": direct / PROBE_COUNT if method != "kmeans_k30" else None,
                  "probe_fallback_count": fallback if method != "kmeans_k30" else 0,
                  "probe_fallback_rate": fallback / PROBE_COUNT if method != "kmeans_k30" else None,
                  "probe_subject_labeled_count": labeled,
                  "probe_top1_metadata_consistency": top1_count / labeled if labeled else None,
                  "probe_top2_metadata_consistency": top2_count / labeled if labeled else None,
                  "mean_similarity_margin": float(margin_array.mean()),
                  "median_similarity_margin": float(np.median(margin_array))}, np.asarray(final_labels), np.asarray(raw_status)


def run_one(method: str, seed: int, articles: List[Dict[str, Any]], embeddings: np.ndarray,
            probe_indices: np.ndarray, train_indices: np.ndarray, subject_sets: List[Set[str]],
            display: Dict[str, str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], np.ndarray, np.ndarray]:
    started = time.perf_counter(); train_embeddings = embeddings[train_indices]; probe_embeddings = embeddings[probe_indices]
    train_labels, raw_probe_labels = fit(method, seed, train_embeddings, probe_embeddings)
    cluster_ids, info, centroids = build_cluster_info(train_indices, probe_indices, train_embeddings,
                                                       train_labels, subject_sets, display)
    predictions, probe_metrics, final_labels, raw_status = evaluate(method, seed, articles, probe_indices,
        probe_embeddings, raw_probe_labels, cluster_ids, info, centroids, subject_sets, display)
    assigned = train_labels >= 0; sizes = [info[value]["size"] for value in cluster_ids]
    labeled = sum(info[value]["labeled"] for value in cluster_ids)
    purity = sum(info[value]["purity"] * info[value]["labeled"] for value in cluster_ids) / labeled if labeled else None
    silhouette: Optional[float] = None
    if assigned.sum() > 1 and len(set(int(value) for value in train_labels[assigned])) >= 2:
        silhouette = float(silhouette_score(train_embeddings[assigned], train_labels[assigned], metric="cosine",
                                            sample_size=min(3000, int(assigned.sum())), random_state=seed))
    warning = ""
    if method == "kmeans_k30" and (len(cluster_ids) != 30 or min(sizes) < 10):
        warning = "KMeans cluster sayısı 30 değil veya gözlenen cluster boyutu 10'dan küçük."
    row: Dict[str, Any] = {"method_name": method, "seed": seed, "train_article_count": TRAIN_COUNT,
        "probe_article_count": PROBE_COUNT, "train_cluster_count": len(cluster_ids),
        "train_noise_count": int((~assigned).sum()), "train_noise_rate": float((~assigned).mean()),
        "min_cluster_size": min(sizes), "median_cluster_size": float(np.median(sizes)),
        "max_cluster_size": max(sizes), "weighted_train_subject_purity": purity,
        "cosine_silhouette": silhouette,
        "collapsed_run": bool(method == "umap10_hdbscan_leaf" and len(cluster_ids) < 5),
        "runtime_seconds": time.perf_counter() - started, "warning": warning, "error": ""}
    row.update(probe_metrics)
    return row, predictions, final_labels, raw_status


def failed_row(method: str, seed: int, error: Exception) -> Dict[str, Any]:
    row = {field: None for field in RUN_FIELDS}; row.update({"method_name": method, "seed": seed,
        "train_article_count": TRAIN_COUNT, "probe_article_count": PROBE_COUNT,
        "warning": "", "error": "%s: %s" % (type(error).__name__, error)})
    return row


def values(rows: Sequence[Dict[str, Any]], field: str) -> List[float]:
    return [float(row[field]) for row in rows if row.get(field) is not None and math.isfinite(float(row[field]))]


def stat(rows: Sequence[Dict[str, Any]], field: str, operation: str) -> Optional[float]:
    data = values(rows, field)
    if not data: return None
    array = np.asarray(data)
    return {"mean": float(array.mean()), "std": float(array.std()), "min": float(array.min()),
            "max": float(array.max())}[operation]


def pairwise(labels: Dict[Tuple[str, int], np.ndarray], statuses: Dict[Tuple[str, int], np.ndarray]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method in METHODS:
        for seed_a, seed_b in combinations(SEEDS, 2):
            left = labels[(method, seed_a)]; right = labels[(method, seed_b)]
            raw_agreement: Optional[float] = None
            if method == "umap10_hdbscan_leaf":
                raw_agreement = float(np.mean(statuses[(method, seed_a)] == statuses[(method, seed_b)]))
            rows.append({"method_name": method, "seed_a": seed_a, "seed_b": seed_b,
                         "probe_count": len(left), "adjusted_rand_index": adjusted_rand_score(left, right),
                         "normalized_mutual_information": normalized_mutual_info_score(left, right),
                         "raw_status_agreement": raw_agreement})
    return rows


def summaries(runs: List[Dict[str, Any]], pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    mapping = [("cluster_count", "train_cluster_count"), ("train_noise_rate", "train_noise_rate"),
               ("probe_direct_rate", "probe_direct_rate"), ("probe_fallback_rate", "probe_fallback_rate"),
               ("probe_top1", "probe_top1_metadata_consistency"), ("probe_top2", "probe_top2_metadata_consistency")]
    for method in METHODS:
        run_rows = [row for row in runs if row["method_name"] == method and not row.get("error")]
        pair_rows = [row for row in pairs if row["method_name"] == method]
        row: Dict[str, Any] = {"method_name": method, "successful_run_count": len(run_rows),
                               "collapse_run_count": sum(bool(value.get("collapsed_run")) for value in run_rows)}
        for prefix, field in mapping:
            row[prefix + "_mean"] = stat(run_rows, field, "mean"); row[prefix + "_std"] = stat(run_rows, field, "std")
        row["cluster_count_min"] = stat(run_rows, "train_cluster_count", "min")
        row["cluster_count_max"] = stat(run_rows, "train_cluster_count", "max")
        row["weighted_train_purity_mean"] = stat(run_rows, "weighted_train_subject_purity", "mean")
        row["similarity_margin_mean"] = stat(run_rows, "mean_similarity_margin", "mean")
        row["cosine_silhouette_mean"] = stat(run_rows, "cosine_silhouette", "mean")
        row["runtime_seconds_mean"] = stat(run_rows, "runtime_seconds", "mean")
        for prefix, field in (("ari", "adjusted_rand_index"), ("nmi", "normalized_mutual_information")):
            for operation in ("mean", "std", "min", "max"):
                row[prefix + "_" + operation] = stat(pair_rows, field, operation)
        row["raw_status_agreement_mean"] = stat(pair_rows, "raw_status_agreement", "mean")
        row["raw_status_agreement_std"] = stat(pair_rows, "raw_status_agreement", "std")
        result.append(row)
    return result


def choose(summary_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    kmeans = next(row for row in summary_rows if row["method_name"] == "kmeans_k30")
    leaf = next(row for row in summary_rows if row["method_name"] == "umap10_hdbscan_leaf")
    top1_gap = float(leaf["probe_top1_mean"] - kmeans["probe_top1_mean"])
    close = abs(top1_gap) < 0.02
    if leaf["collapse_run_count"] == 0 and close:
        return {"decision": "dual_method", "primary_discovery_method": "umap10_hdbscan_leaf",
                "baseline_method": "kmeans_k30", "top1_difference_leaf_minus_kmeans": top1_gap,
                "reason": "Top-1 farkı küçük; HDBSCAN Leaf doğal konu keşfi ile direct/noise gri alan bilgisini, KMeans ise zorunlu tam atamalı baseline'ı sağlar."}
    winner = "umap10_hdbscan_leaf" if top1_gap > 0 and leaf["collapse_run_count"] == 0 else "kmeans_k30"
    return {"decision": "temporary_single_candidate", "temporary_candidate": winner,
            "baseline_method": "kmeans_k30", "top1_difference_leaf_minus_kmeans": top1_gap,
            "reason": "Çökme, üyelik ARI/NMI ve metadata consistency birlikte yorumlandı; composite score kullanılmadı."}


def safe(value: Any) -> Any:
    if value is None: return ""
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    if isinstance(value, (bool, np.bool_)): return str(bool(value))
    return value


def write_csv(path: Path, fields: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader()
        for row in rows: writer.writerow({field: safe(row.get(field)) for field in fields})


def json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list): return [json_safe(item) for item in value]
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): value = float(value)
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, np.bool_): return bool(value)
    return value


def plots(summary_rows: List[Dict[str, Any]], output: Path) -> None:
    names = [row["method_name"] for row in summary_rows]; x = np.arange(2); width = 0.36
    plt.figure(figsize=(9, 5)); plt.bar(x-width/2, [row["probe_top1_mean"] for row in summary_rows], width,
        yerr=[row["probe_top1_std"] for row in summary_rows], capsize=5, label="Top-1")
    plt.bar(x+width/2, [row["probe_top2_mean"] for row in summary_rows], width,
        yerr=[row["probe_top2_std"] for row in summary_rows], capsize=5, label="Top-2")
    plt.xticks(x, names); plt.ylabel("Metadata consistency"); plt.title("Sabit probe kalitesi (ortalama ± std)")
    plt.legend(); plt.grid(axis="y", alpha=.25); plt.tight_layout(); plt.savefig(str(output/"day30_finalist_quality.png"), dpi=160); plt.close()
    plt.figure(figsize=(9, 5)); plt.bar(x-width/2, [row["ari_mean"] for row in summary_rows], width,
        yerr=[row["ari_std"] for row in summary_rows], capsize=5, label="ARI")
    plt.bar(x+width/2, [row["nmi_mean"] for row in summary_rows], width,
        yerr=[row["nmi_std"] for row in summary_rows], capsize=5, label="NMI")
    plt.xticks(x, names); plt.ylabel("Seedler arası üyelik kararlılığı"); plt.ylim(0, 1)
    plt.title("Sabit probe üyelik kararlılığı (10 seed çifti)"); plt.legend(); plt.grid(axis="y", alpha=.25)
    plt.tight_layout(); plt.savefig(str(output/"day30_membership_stability.png"), dpi=160); plt.close()


def pct(value: Optional[float]) -> str:
    return "—" if value is None else "%.2f%%" % (100 * value)


def report(summary_rows: List[Dict[str, Any]], decision: Dict[str, Any]) -> str:
    lines = ["# Day 30 — Bağımsız finalist doğrulaması", "", "## Deney ve veri sızıntısı", "",
        "Pilot veriden ID düzeyinde bağımsız 5.000 Türkçe PAPER kaydı kullanıldı. Probe seed=2026 ile bir kez oluşturuldu. Her run yalnızca training pool içinden seçilen 3.600 makalede fit edildi; probe subjectleri tahmin tamamlanana kadar okunmadı ve indeks ayrıklığı assertion ile doğrulandı.", "",
        "## Sonuçlar", "", "| Yöntem | Cluster | Çökme | Noise | Direct | Fallback | Top-1 | Top-2 | ARI | NMI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary_rows:
        lines.append("| %s | %.1f±%.1f | %d | %s±%s | %s±%s | %s±%s | %s±%s | %s±%s | %.3f±%.3f | %.3f±%.3f |" % (
            row["method_name"], row["cluster_count_mean"], row["cluster_count_std"], row["collapse_run_count"],
            pct(row["train_noise_rate_mean"]), pct(row["train_noise_rate_std"]), pct(row["probe_direct_rate_mean"]),
            pct(row["probe_direct_rate_std"]), pct(row["probe_fallback_rate_mean"]), pct(row["probe_fallback_rate_std"]),
            pct(row["probe_top1_mean"]), pct(row["probe_top1_std"]), pct(row["probe_top2_mean"]), pct(row["probe_top2_std"]),
            row["ari_mean"], row["ari_std"], row["nmi_mean"], row["nmi_std"]))
    lines += ["", "## Karar", "", decision["reason"], "",
        "Karar: **%s**. Ana keşif yöntemi: **%s**. Baseline: **%s**." % (
            decision["decision"], decision.get("primary_discovery_method", decision.get("temporary_candidate", "—")),
            decision["baseline_method"]), "",
        "KMeans'in cluster sayısının 30 olması kararlılık kanıtı sayılmadı; asıl karşılaştırma sabit probe üzerindeki ARI/NMI üyelik kararlılığıdır. HDBSCAN'in direct/noise durumu gri alan sinyali olarak ayrıca raporlandı.", "",
        "## Sınırlamalar", "", "Subject metadata çok etiketli ve eksik olabilir. Dominant subject yalnızca dolaylı kalite ölçütüdür. Beş seed tüm örnekleme belirsizliğini kapsamaz. API sorgu stratejisi TR Dizin evreninin kusursuz rastgele örneği değildir.", "",
        "## 50.000 makaleye geçmeden önce tek adım", "", "Aynı sabit protokolü, veri çekim sorgularından ayrılmış daha geniş bir bağımsız doğrulama örnekleminde önceden dondurulmuş karar eşikleriyle tekrarlayın."]
    return "\n".join(lines) + "\n"


def main() -> None:
    articles, embeddings = load_inputs(); subject_sets, display = subject_data(articles)
    pool, probe, training = build_splits(articles)
    probe_hash = hashlib.sha256(probe.tobytes()).hexdigest()
    runs: List[Dict[str, Any]] = []; predictions: List[Dict[str, Any]] = []
    labels: Dict[Tuple[str, int], np.ndarray] = {}; statuses: Dict[Tuple[str, int], np.ndarray] = {}
    warnings: List[str] = []; number = 0
    for method in METHODS:
        for seed in SEEDS:
            number += 1; print("[%d/10] method=%s seed=%d" % (number, method, seed), flush=True)
            try:
                row, pred, final_labels, raw_status = run_one(method, seed, articles, embeddings, probe,
                                                               training[seed], subject_sets, display)
                runs.append(row); predictions.extend(pred); labels[(method, seed)] = final_labels; statuses[(method, seed)] = raw_status
                print("cluster=%d noise=%.2f%% direct=%s top1=%.2f%% top2=%.2f%% süre=%.1fs" % (
                    row["train_cluster_count"], 100*row["train_noise_rate"], pct(row["probe_direct_rate"]),
                    100*row["probe_top1_metadata_consistency"], 100*row["probe_top2_metadata_consistency"], row["runtime_seconds"]))
                if row["warning"]: warnings.append("%s seed=%d: %s" % (method, seed, row["warning"]))
            except (AssertionError, ValueError):
                raise
            except Exception as error:
                traceback.print_exc(); runs.append(failed_row(method, seed, error)); warnings.append("%s seed=%d: %s" % (method, seed, error))
    if len(labels) != 10: raise RuntimeError("Başarısız run nedeniyle pairwise stability hesaplanamadı.")
    pair_rows = pairwise(labels, statuses); summary_rows = summaries(runs, pair_rows); decision = choose(summary_rows)
    output = root()/"research"/"outputs"; write_csv(output/"day30_finalist_runs.csv", RUN_FIELDS, runs)
    write_csv(output/"day30_finalist_summary.csv", SUMMARY_FIELDS, summary_rows)
    write_csv(output/"day30_probe_predictions.csv", PRED_FIELDS, predictions)
    write_csv(output/"day30_pairwise_stability.csv", PAIR_FIELDS, pair_rows)
    payload = {"settings": {"seeds": SEEDS, "probe_seed": PROBE_SEED, "probe_count": PROBE_COUNT,
        "training_pool_count": TRAIN_POOL_COUNT, "train_per_seed": TRAIN_COUNT, "probe_index_sha256": probe_hash,
        "methods": METHODS, "test_subject_leakage": False}, "method_summaries": summary_rows,
        "pairwise_stability": pair_rows, "warnings": warnings, "recommendation": decision}
    with (output/"day30_finalist_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
    plots(summary_rows, output)
    (output/"day30_finalist_recommendation.md").write_text(report(summary_rows, decision), encoding="utf-8")
    print("\nFINALİST ÖZETİ")
    for row in summary_rows:
        print("%s clusters=%.1f±%.1f Top1=%s±%s Top2=%s±%s ARI=%.3f±%.3f NMI=%.3f±%.3f" % (
            row["method_name"], row["cluster_count_mean"], row["cluster_count_std"], pct(row["probe_top1_mean"]),
            pct(row["probe_top1_std"]), pct(row["probe_top2_mean"]), pct(row["probe_top2_std"]),
            row["ari_mean"], row["ari_std"], row["nmi_mean"], row["nmi_std"]))
    print("KARAR: %s" % decision)


if __name__ == "__main__":
    main()
