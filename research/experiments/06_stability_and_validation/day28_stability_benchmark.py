#!/usr/bin/env python3
"""Day 28: çoklu seed clustering kararlılığı ve holdout kalite benchmarkı."""

import csv
import json
import math
import os
import traceback
from collections import Counter
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
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split


SEEDS = [11, 22, 33, 42, 55]
ARTICLE_COUNT = 1000
TRAIN_COUNT = 800
TEST_COUNT = 200

METHODS = [
    {"name": "umap10_hdbscan_eom", "reducer": "umap", "clusterer": "hdbscan", "selection": "eom"},
    {"name": "umap10_hdbscan_leaf", "reducer": "umap", "clusterer": "hdbscan", "selection": "leaf"},
    {"name": "pca50_hdbscan_eom", "reducer": "pca", "clusterer": "hdbscan", "selection": "eom"},
    {"name": "pca50_hdbscan_leaf", "reducer": "pca", "clusterer": "hdbscan", "selection": "leaf"},
    {"name": "kmeans_k30_baseline", "reducer": "none", "clusterer": "kmeans", "selection": "kmeans"},
]

RUN_FIELDS = [
    "method_name", "seed", "train_article_count", "test_article_count",
    "train_cluster_count", "train_noise_count", "train_noise_rate",
    "train_assigned_count", "min_cluster_size_observed",
    "median_cluster_size_observed", "max_cluster_size_observed",
    "train_subject_labeled_count", "weighted_train_subject_purity",
    "test_direct_hdbscan_count", "test_direct_hdbscan_rate",
    "test_centroid_fallback_count", "test_centroid_fallback_rate",
    "test_subject_labeled_count", "holdout_top1_metadata_consistency",
    "holdout_top2_metadata_consistency", "direct_top1_metadata_consistency",
    "direct_top2_metadata_consistency", "fallback_top1_metadata_consistency",
    "fallback_top2_metadata_consistency", "mean_primary_similarity",
    "mean_secondary_similarity", "mean_similarity_margin",
    "median_similarity_margin", "cosine_silhouette", "collapsed_run",
    "warning", "error",
]

PREDICTION_FIELDS = [
    "method_name", "seed", "test_position", "global_row_index", "article_id",
    "publication_year", "title_tr", "direct_or_fallback",
    "raw_predicted_cluster", "primary_cluster", "primary_topic",
    "primary_similarity", "secondary_cluster", "secondary_topic",
    "secondary_similarity", "similarity_margin", "known_subjects",
    "top1_matches_metadata", "top2_matches_metadata",
]

SUMMARY_FIELDS = [
    "method_name", "cluster_count_mean", "cluster_count_std", "cluster_count_min",
    "cluster_count_max", "cluster_count_coefficient_of_variation",
    "collapse_run_count", "train_noise_rate_mean", "train_noise_rate_std",
    "holdout_top1_mean", "holdout_top1_std", "holdout_top1_min",
    "holdout_top1_max", "holdout_top2_mean", "holdout_top2_std",
    "direct_rate_mean", "fallback_rate_mean",
    "weighted_train_subject_purity_mean", "cosine_silhouette_mean",
    "successful_run_count",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding matrisinde sıfır uzunluklu vektör var.")
    return values / norms


def load_inputs() -> Tuple[List[Dict[str, Any]], np.ndarray]:
    article_path = project_root() / "data/processed/pilot_articles_1000.jsonl"
    embedding_path = project_root() / "research/outputs/day13_embeddings/tr_mteb.npy"
    articles: List[Dict[str, Any]] = []
    with article_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("JSONL satırı nesne değil: %d" % line_number)
                articles.append(value)
    embeddings = np.load(str(embedding_path))
    if len(articles) != ARTICLE_COUNT or embeddings.shape[0] != ARTICLE_COUNT:
        raise ValueError("Makale/embedding sayısı 1000 değil: %d / %s" % (len(articles), embeddings.shape))
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("Embedding matrisi 2D değil veya NaN/sonsuz değer içeriyor.")
    ids = [str(article.get("article_id", "")) for article in articles]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("article_id alanları boş veya benzersiz değil.")
    return articles, normalize_rows(embeddings)


def parse_subject(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def subject_key(subject: Dict[str, Any]) -> Optional[str]:
    if subject.get("id") is not None:
        return "id:%s" % subject["id"]
    for field in ("fullName", "name"):
        value = subject.get(field)
        if isinstance(value, str) and value.strip():
            return "%s:%s" % (field, value.strip())
    return None


def build_subjects(articles: List[Dict[str, Any]]) -> Tuple[List[Set[str]], Dict[str, str]]:
    all_sets: List[Set[str]] = []
    display: Dict[str, str] = {}
    for article in articles:
        keys: Set[str] = set()
        raw_values = article.get("subjects", [])
        if not isinstance(raw_values, list):
            raw_values = []
        for raw in raw_values:
            parsed = parse_subject(raw)
            if parsed is None:
                continue
            key = subject_key(parsed)
            if key is None:
                continue
            keys.add(key)
            shown = parsed.get("fullName") or parsed.get("name") or key
            display[key] = str(shown).strip()
        all_sets.append(keys)
    return all_sets, display


def split_indices(articles: List[Dict[str, Any]], seed: int) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(articles), dtype=np.int32)
    years = np.asarray([str(article.get("publication_year", "unknown")) for article in articles])
    train, test = train_test_split(indices, test_size=TEST_COUNT, random_state=seed, stratify=years)
    train = np.sort(train.astype(np.int32, copy=False))
    test = np.sort(test.astype(np.int32, copy=False))
    if len(train) != TRAIN_COUNT or len(test) != TEST_COUNT or np.intersect1d(train, test).size:
        raise AssertionError("Geçersiz train/test split.")
    return train, test


def reduce_data(method: Dict[str, str], train: np.ndarray, test: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if method["reducer"] == "umap":
        reducer = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0,
                            metric="cosine", random_state=seed, low_memory=True)
    elif method["reducer"] == "pca":
        reducer = PCA(n_components=50, svd_solver="randomized", random_state=seed)
    else:
        return train, test
    train_reduced = reducer.fit_transform(train)
    test_reduced = reducer.transform(test)
    return train_reduced, test_reduced


def fit_clusters(method: Dict[str, str], train_reduced: np.ndarray,
                 test_reduced: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if method["clusterer"] == "kmeans":
        model = KMeans(n_clusters=30, n_init=20, random_state=seed)
        train_labels = model.fit_predict(train_reduced)
        return train_labels.astype(np.int32), model.predict(test_reduced).astype(np.int32)
    model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric="euclidean",
                            cluster_selection_method=method["selection"],
                            prediction_data=True, gen_min_span_tree=False)
    train_labels = model.fit_predict(train_reduced).astype(np.int32)
    test_labels, unused_strengths = approximate_predict(model, test_reduced)
    return train_labels, test_labels.astype(np.int32)


def build_cluster_info(train_indices: np.ndarray, test_indices: np.ndarray,
                       train_embeddings: np.ndarray, train_labels: np.ndarray,
                       subject_sets: List[Set[str]], display: Dict[str, str]) -> Tuple[List[int], Dict[int, Dict[str, Any]], np.ndarray]:
    # Subject metadata erişiminin test kümesine taşmadığını açıkça doğrula.
    accessed_indices = set(int(value) for value in train_indices)
    assert accessed_indices.isdisjoint(set(int(value) for value in test_indices))
    cluster_ids = sorted(set(int(value) for value in train_labels if int(value) >= 0))
    if not cluster_ids:
        raise RuntimeError("Eğitim verisinde cluster oluşmadı.")
    info: Dict[int, Dict[str, Any]] = {}
    centroids: List[np.ndarray] = []
    for cluster_id in cluster_ids:
        local = np.where(train_labels == cluster_id)[0]
        global_indices = [int(train_indices[int(value)]) for value in local]
        assert set(global_indices).issubset(accessed_indices)
        centroid = train_embeddings[local].mean(axis=0, keepdims=True)
        centroids.append(normalize_rows(centroid)[0])
        counter: Counter = Counter()
        labeled = 0
        for global_index in global_indices:
            values = subject_sets[global_index]
            if values:
                labeled += 1
                counter.update(sorted(values))  # deterministic; subject/makale başına en çok bir kez.
        top = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:3]
        dominant = top[0][0] if top else ""
        dominant_count = int(top[0][1]) if top else 0
        info[cluster_id] = {
            "size": len(local), "labeled": labeled, "dominant_key": dominant,
            "dominant_name": display.get(dominant, "Cluster %d" % cluster_id),
            "top_subjects": [{"key": key, "name": display.get(key, key), "count": int(count)} for key, count in top],
            "purity": float(dominant_count / labeled) if labeled else 0.0,
        }
    assert accessed_indices.isdisjoint(set(int(value) for value in test_indices))
    return cluster_ids, info, np.vstack(centroids).astype(np.float32)


def safe_rate(numerator: int, denominator: int) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def evaluate_predictions(method: Dict[str, str], seed: int, articles: List[Dict[str, Any]],
                         test_indices: np.ndarray, test_embeddings: np.ndarray,
                         raw_labels: np.ndarray, cluster_ids: List[int],
                         info: Dict[int, Dict[str, Any]], centroids: np.ndarray,
                         subject_sets: List[Set[str]], display: Dict[str, str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    scores = test_embeddings @ centroids.T
    positions = {cluster_id: position for position, cluster_id in enumerate(cluster_ids)}
    rows: List[Dict[str, Any]] = []
    buckets = {"all": [0, 0, 0], "direct": [0, 0, 0], "fallback": [0, 0, 0]}
    similarities: List[Tuple[float, float, float]] = []
    direct = 0
    fallback = 0
    for test_position, raw_global_index in enumerate(test_indices):
        global_index = int(raw_global_index)
        raw_label = int(raw_labels[test_position])
        ranked = [cluster_ids[int(pos)] for pos in np.argsort(scores[test_position])[::-1]]
        if raw_label >= 0:
            primary = raw_label
            mode = "direct"
            direct += 1
        else:
            primary = ranked[0]
            mode = "fallback"
            fallback += 1
        secondary = next((value for value in ranked if value != primary), None)
        if secondary is None:
            secondary = primary
        primary_similarity = float(scores[test_position, positions[primary]])
        secondary_similarity = float(scores[test_position, positions[secondary]])
        margin = primary_similarity - secondary_similarity
        similarities.append((primary_similarity, secondary_similarity, margin))
        known = subject_sets[global_index]
        if known:
            top1 = bool(info[primary]["dominant_key"] and info[primary]["dominant_key"] in known)
            top2 = bool(top1 or (info[secondary]["dominant_key"] and info[secondary]["dominant_key"] in known))
            for bucket in (buckets["all"], buckets[mode]):
                bucket[0] += 1
                bucket[1] += int(top1)
                bucket[2] += int(top2)
        else:
            top1 = None
            top2 = None
        article = articles[global_index]
        assert str(article.get("article_id", "")) == str(articles[int(test_indices[test_position])].get("article_id", ""))
        rows.append({
            "method_name": method["name"], "seed": seed, "test_position": test_position,
            "global_row_index": global_index, "article_id": article.get("article_id", ""),
            "publication_year": article.get("publication_year", ""), "title_tr": article.get("title_tr", ""),
            "direct_or_fallback": mode, "raw_predicted_cluster": raw_label,
            "primary_cluster": primary, "primary_topic": info[primary]["dominant_name"],
            "primary_similarity": primary_similarity, "secondary_cluster": secondary,
            "secondary_topic": info[secondary]["dominant_name"], "secondary_similarity": secondary_similarity,
            "similarity_margin": margin,
            "known_subjects": " | ".join(sorted(display.get(key, key) for key in known)),
            "top1_matches_metadata": top1, "top2_matches_metadata": top2,
        })
    array = np.asarray(similarities, dtype=np.float64)
    is_kmeans = method["clusterer"] == "kmeans"
    return rows, {
        "test_direct_hdbscan_count": 0 if is_kmeans else direct,
        "test_direct_hdbscan_rate": 0.0 if is_kmeans else direct / len(test_indices),
        "test_centroid_fallback_count": 0 if is_kmeans else fallback,
        "test_centroid_fallback_rate": 0.0 if is_kmeans else fallback / len(test_indices),
        "test_subject_labeled_count": buckets["all"][0],
        "holdout_top1_metadata_consistency": safe_rate(buckets["all"][1], buckets["all"][0]),
        "holdout_top2_metadata_consistency": safe_rate(buckets["all"][2], buckets["all"][0]),
        "direct_top1_metadata_consistency": safe_rate(buckets["direct"][1], buckets["direct"][0]) if not is_kmeans else None,
        "direct_top2_metadata_consistency": safe_rate(buckets["direct"][2], buckets["direct"][0]) if not is_kmeans else None,
        "fallback_top1_metadata_consistency": safe_rate(buckets["fallback"][1], buckets["fallback"][0]) if not is_kmeans else None,
        "fallback_top2_metadata_consistency": safe_rate(buckets["fallback"][2], buckets["fallback"][0]) if not is_kmeans else None,
        "mean_primary_similarity": float(array[:, 0].mean()),
        "mean_secondary_similarity": float(array[:, 1].mean()),
        "mean_similarity_margin": float(array[:, 2].mean()),
        "median_similarity_margin": float(np.median(array[:, 2])),
    }


def run_one(method: Dict[str, str], seed: int, articles: List[Dict[str, Any]],
            embeddings: np.ndarray, subject_sets: List[Set[str]],
            display: Dict[str, str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    train_indices, test_indices = split_indices(articles, seed)
    train_embeddings = embeddings[train_indices]
    test_embeddings = embeddings[test_indices]
    train_reduced, test_reduced = reduce_data(method, train_embeddings, test_embeddings, seed)
    train_labels, test_labels = fit_clusters(method, train_reduced, test_reduced, seed)
    cluster_ids, info, centroids = build_cluster_info(
        train_indices, test_indices, train_embeddings, train_labels, subject_sets, display)
    prediction_rows, test_metrics = evaluate_predictions(
        method, seed, articles, test_indices, test_embeddings, test_labels,
        cluster_ids, info, centroids, subject_sets, display)
    sizes = [info[value]["size"] for value in cluster_ids]
    assigned_mask = train_labels >= 0
    assigned_count = int(assigned_mask.sum())
    noise_count = int(len(train_labels) - assigned_count)
    labeled_count = sum(info[value]["labeled"] for value in cluster_ids)
    weighted_purity = (sum(info[value]["purity"] * info[value]["labeled"] for value in cluster_ids) / labeled_count
                       if labeled_count else None)
    silhouette: Optional[float] = None
    assigned_labels = train_labels[assigned_mask]
    if assigned_count >= 2 and len(set(int(value) for value in assigned_labels)) >= 2:
        silhouette = float(silhouette_score(train_embeddings[assigned_mask], assigned_labels, metric="cosine"))
    warning = ""
    if method["clusterer"] == "kmeans" and (min(sizes) < 5 or max(sizes) > 0.50 * len(train_labels)):
        warning = "KMeans boş veya aşırı küçük/büyük cluster uyarısı (min<5 veya max>%%50)."
    result: Dict[str, Any] = {
        "method_name": method["name"], "seed": seed, "train_article_count": len(train_indices),
        "test_article_count": len(test_indices), "train_cluster_count": len(cluster_ids),
        "train_noise_count": noise_count, "train_noise_rate": noise_count / len(train_labels),
        "train_assigned_count": assigned_count, "min_cluster_size_observed": min(sizes),
        "median_cluster_size_observed": float(np.median(sizes)), "max_cluster_size_observed": max(sizes),
        "train_subject_labeled_count": labeled_count, "weighted_train_subject_purity": weighted_purity,
        "cosine_silhouette": silhouette,
        "collapsed_run": bool(method["clusterer"] == "hdbscan" and len(cluster_ids) < 5),
        "warning": warning, "error": "",
    }
    result.update(test_metrics)
    return result, prediction_rows


def failed_row(method: Dict[str, str], seed: int, error: Exception) -> Dict[str, Any]:
    row = {field: None for field in RUN_FIELDS}
    row.update({"method_name": method["name"], "seed": seed, "train_article_count": TRAIN_COUNT,
                "test_article_count": TEST_COUNT, "collapsed_run": None, "warning": "",
                "error": "%s: %s" % (type(error).__name__, error)})
    return row


def finite_values(rows: Sequence[Dict[str, Any]], field: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(field)
        if value is not None and isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def statistic(rows: Sequence[Dict[str, Any]], field: str, operation: str) -> Optional[float]:
    values = finite_values(rows, field)
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if operation == "mean": return float(array.mean())
    if operation == "std": return float(array.std(ddof=0))
    if operation == "min": return float(array.min())
    if operation == "max": return float(array.max())
    raise ValueError(operation)


def summarize(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for method in METHODS:
        rows = [row for row in runs if row["method_name"] == method["name"] and not row.get("error")]
        cluster_mean = statistic(rows, "train_cluster_count", "mean")
        summary = {
            "method_name": method["name"], "cluster_count_mean": cluster_mean,
            "cluster_count_std": statistic(rows, "train_cluster_count", "std"),
            "cluster_count_min": statistic(rows, "train_cluster_count", "min"),
            "cluster_count_max": statistic(rows, "train_cluster_count", "max"),
            "cluster_count_coefficient_of_variation": (statistic(rows, "train_cluster_count", "std") / cluster_mean
                                                       if cluster_mean else None),
            "collapse_run_count": sum(bool(row.get("collapsed_run")) for row in rows),
            "train_noise_rate_mean": statistic(rows, "train_noise_rate", "mean"),
            "train_noise_rate_std": statistic(rows, "train_noise_rate", "std"),
            "holdout_top1_mean": statistic(rows, "holdout_top1_metadata_consistency", "mean"),
            "holdout_top1_std": statistic(rows, "holdout_top1_metadata_consistency", "std"),
            "holdout_top1_min": statistic(rows, "holdout_top1_metadata_consistency", "min"),
            "holdout_top1_max": statistic(rows, "holdout_top1_metadata_consistency", "max"),
            "holdout_top2_mean": statistic(rows, "holdout_top2_metadata_consistency", "mean"),
            "holdout_top2_std": statistic(rows, "holdout_top2_metadata_consistency", "std"),
            "direct_rate_mean": statistic(rows, "test_direct_hdbscan_rate", "mean"),
            "fallback_rate_mean": statistic(rows, "test_centroid_fallback_rate", "mean"),
            "weighted_train_subject_purity_mean": statistic(rows, "weighted_train_subject_purity", "mean"),
            "cosine_silhouette_mean": statistic(rows, "cosine_silhouette", "mean"),
            "successful_run_count": len(rows),
        }
        summaries.append(summary)
    return summaries


def choose_recommendation(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    stable = [row for row in summaries if row["successful_run_count"] == len(SEEDS)
              and row["collapse_run_count"] == 0 and row["cluster_count_min"] is not None
              and row["cluster_count_min"] >= 5]
    excluded = [row["method_name"] for row in summaries if row not in stable]
    if not stable:
        return {"status": "karar_verilemedi", "temporary_candidate": None, "excluded_methods": excluded,
                "reason": "Beş seedin tamamında başarılı ve anlamlı cluster yapısı üreten yöntem yok."}
    ranked = sorted(stable, key=lambda row: (-float(row["holdout_top1_mean"] or -1),
                                             float(row["holdout_top1_std"] or 1),
                                             -float(row["holdout_top2_mean"] or -1)))
    candidate = ranked[0]
    baseline = next(row for row in summaries if row["method_name"] == "kmeans_k30_baseline")
    reason = ("Çökmesiz yöntemler arasında Holdout Top-1 ortalaması, ardından düşük standart sapma "
              "ve Top-2 ortalaması kullanılarak geçici aday belirlendi; composite score kullanılmadı.")
    return {"status": "gecici_aday", "temporary_candidate": candidate["method_name"],
            "excluded_methods": excluded, "reason": reason,
            "kmeans_comparison": {"kmeans_top1_mean": baseline["holdout_top1_mean"],
                                  "candidate_top1_mean": candidate["holdout_top1_mean"],
                                  "kmeans_more_stable_by_cluster_cv":
                                      (baseline["cluster_count_coefficient_of_variation"] or 0) <
                                      (candidate["cluster_count_coefficient_of_variation"] or 0)}}


def csv_value(value: Any) -> Any:
    if value is None: return ""
    if isinstance(value, (np.bool_, bool)): return str(bool(value))
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    return value


def write_csv(path: Path, fields: List[str], rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): value = float(value)
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, np.bool_): return bool(value)
    return value


def create_plots(runs: List[Dict[str, Any]], summaries: List[Dict[str, Any]], output: Path) -> None:
    plt.figure(figsize=(12, 6))
    for method in METHODS:
        rows = sorted([row for row in runs if row["method_name"] == method["name"] and not row.get("error")], key=lambda row: row["seed"])
        plt.plot([row["seed"] for row in rows], [row["train_cluster_count"] for row in rows], marker="o", label=method["name"])
    plt.xlabel("Seed"); plt.ylabel("Eğitim cluster sayısı"); plt.title("Cluster sayısı kararlılığı")
    plt.xticks(SEEDS); plt.grid(alpha=0.25); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(str(output / "day28_cluster_count_stability.png"), dpi=160); plt.close()

    names = [row["method_name"] for row in summaries]
    x_values = np.arange(len(names)); width = 0.36
    top1 = [row["holdout_top1_mean"] or 0 for row in summaries]
    top2 = [row["holdout_top2_mean"] or 0 for row in summaries]
    top1_std = [row["holdout_top1_std"] or 0 for row in summaries]
    top2_std = [statistic([run for run in runs if run["method_name"] == name and not run.get("error")],
                          "holdout_top2_metadata_consistency", "std") or 0 for name in names]
    plt.figure(figsize=(12, 6))
    plt.bar(x_values - width / 2, top1, width, yerr=top1_std, capsize=4, label="Holdout Top-1")
    plt.bar(x_values + width / 2, top2, width, yerr=top2_std, capsize=4, label="Holdout Top-2")
    plt.xticks(x_values, names, rotation=20, ha="right"); plt.ylabel("Metadata consistency")
    plt.ylim(0, max(top2 + [0.1]) * 1.25); plt.title("Holdout metadata consistency (ortalama ± std)")
    plt.legend(); plt.grid(axis="y", alpha=0.25); plt.tight_layout()
    plt.savefig(str(output / "day28_holdout_quality.png"), dpi=160); plt.close()


def fmt(value: Optional[float], percent: bool = False) -> str:
    if value is None: return "—"
    return ("%.2f%%" % (100 * value)) if percent else ("%.2f" % value)


def create_report(summaries: List[Dict[str, Any]], runs: List[Dict[str, Any]], recommendation: Dict[str, Any]) -> str:
    lines = ["# Day 28 — Clustering Kararlılık Benchmarkı", "", "## Deney amacı", "",
             "Tek bir 800/200 bölmesine bağımlı yöntem seçimini, beş stratified seed üzerinde karşılaştırmak.", "",
             "## Veri sızıntısının engellenmesi", "",
             "Reducer ve cluster modeli yalnızca 800 eğitim embeddinginde fit edildi. Cluster konu adları yalnızca eğitim subject metadata’sından üretildi; test subjectleri yalnızca tahmin tamamlandıktan sonra metadata consistency değerlendirmesinde kullanıldı. Train/test indeks ayrıklığı assertion ile doğrulandı.", "",
             "## Seed bazlı davranış", "",
             "| Yöntem | Seed sonuçları (cluster/noise%/Top-1%/Top-2%) |", "|---|---|"]
    for method in METHODS:
        items = []
        for row in [value for value in runs if value["method_name"] == method["name"]]:
            if row.get("error"): items.append("%s: HATA" % row["seed"])
            else: items.append("%s: %d/%.1f/%.1f/%.1f" % (row["seed"], row["train_cluster_count"],
                               100 * row["train_noise_rate"], 100 * row["holdout_top1_metadata_consistency"],
                               100 * row["holdout_top2_metadata_consistency"]))
        lines.append("| %s | %s |" % (method["name"], "; ".join(items)))
    lines += ["", "## Yöntem özeti", "", "| Yöntem | Cluster ort±std | Min–max | Çökme | Noise ort±std | Top-1 ort±std | Top-2 ort±std | Train purity | Silhouette |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append("| %s | %s±%s | %s–%s | %d | %s±%s | %s±%s | %s±%s | %s | %s |" % (
            row["method_name"], fmt(row["cluster_count_mean"]), fmt(row["cluster_count_std"]),
            fmt(row["cluster_count_min"]), fmt(row["cluster_count_max"]), row["collapse_run_count"],
            fmt(row["train_noise_rate_mean"], True), fmt(row["train_noise_rate_std"], True),
            fmt(row["holdout_top1_mean"], True), fmt(row["holdout_top1_std"], True),
            fmt(row["holdout_top2_mean"], True), fmt(row["holdout_top2_std"], True),
            fmt(row["weighted_train_subject_purity_mean"], True), fmt(row["cosine_silhouette_mean"])))
    collapsed = [row["method_name"] for row in summaries if row["collapse_run_count"] > 0]
    baseline = next(row for row in summaries if row["method_name"] == "kmeans_k30_baseline")
    candidate = next((row for row in summaries
                      if row["method_name"] == recommendation["temporary_candidate"]), None)
    gaps = []
    for row in summaries:
        if row["weighted_train_subject_purity_mean"] is not None and row["holdout_top1_mean"] is not None:
            gap = row["weighted_train_subject_purity_mean"] - row["holdout_top1_mean"]
            gaps.append("%s: %+.2f puan" % (row["method_name"], 100 * gap))
    if candidate is not None:
        baseline_text = ("Geçici adayın KMeans’e göre Top-1 farkı %+.2f, Top-2 farkı %+.2f yüzde puandır. "
                         "Cluster sayısı CV değerleri sırasıyla %.3f ve %.3f’tür." % (
                             100 * (candidate["holdout_top1_mean"] - baseline["holdout_top1_mean"]),
                             100 * (candidate["holdout_top2_mean"] - baseline["holdout_top2_mean"]),
                             candidate["cluster_count_coefficient_of_variation"],
                             baseline["cluster_count_coefficient_of_variation"]))
    else:
        baseline_text = "Kararlı bir geçici aday olmadığı için KMeans’e karşı sayısal üstünlük iddiası kurulmadı."
    lines += ["", "## Karar", "", "Cluster çökmesi görülen yöntemler: %s." % (", ".join(collapsed) if collapsed else "yok"),
              "", "Elenen yöntemler: %s." % (", ".join(recommendation["excluded_methods"]) if recommendation["excluded_methods"] else "yok"),
              "", "Geçici aday: **%s**." % (recommendation["temporary_candidate"] or "belirlenemedi"), "",
              recommendation["reason"], "",
              baseline_text, "",
              "Train purity − Holdout Top-1 farkları: %s. Büyük pozitif fark, eğitim metadata uyumunun holdout’a taşınmadığına işaret eder." % "; ".join(gaps), "",
              "Noise oranının seed değişimi tabloda ortalama±standart sapma olarak verildi. KMeans baseline sonucu saklanmadan aynı holdout ölçütleriyle karşılaştırılmıştır. Train purity, holdout başarısı yerine geçmez; silhouette seçim ölçütü olarak tek başına kullanılmamıştır.", "",
              "## 50.000 makaleden önce önerilen sonraki adım", "",
              "Geçici adayı ve KMeans baseline’ı, bağımsız ve daha büyük bir doğrulama örnekleminde aynı sabit protokolle tekrar karşılaştırın.", "",
              "## Sınırlamalar", "",
              "Subject metadata çok etiketli ve eksik olabilir; dominant subject yaklaşımı konu kalitesinin yalnızca dolaylı bir göstergesidir. Beş seed belirsizliği azaltır ancak tüm olası örneklem değişimini kapsamaz. UMAP dönüşümü ve approximate_predict sonuçları kullanılan kütüphane/sürüm ve çalışma ortamına bağlı olabilir."]
    return "\n".join(lines) + "\n"


def print_summary(summaries: List[Dict[str, Any]], recommendation: Dict[str, Any]) -> None:
    print("\nMETHOD KARARLILIK ÖZETİ")
    print("Yöntem | Cluster ort±std | Cluster min-max | Collapse | Noise ort±std | Top-1 ort±std | Top-2 ort±std")
    for row in summaries:
        print("%s | %s±%s | %s-%s | %d | %s±%s | %s±%s | %s±%s" % (
            row["method_name"], fmt(row["cluster_count_mean"]), fmt(row["cluster_count_std"]),
            fmt(row["cluster_count_min"]), fmt(row["cluster_count_max"]), row["collapse_run_count"],
            fmt(row["train_noise_rate_mean"], True), fmt(row["train_noise_rate_std"], True),
            fmt(row["holdout_top1_mean"], True), fmt(row["holdout_top1_std"], True),
            fmt(row["holdout_top2_mean"], True), fmt(row["holdout_top2_std"], True)))
    print("\nGEÇİCİ YÖNTEM KARARI")
    print("- Elenen yöntemler: %s" % (", ".join(recommendation["excluded_methods"]) or "yok"))
    print("- Geçici en iyi aday: %s" % (recommendation["temporary_candidate"] or "belirlenemedi"))
    print("- KMeans baseline ile karşılaştırma: JSON ve Markdown raporda ayrıntılı olarak kaydedildi.")
    if recommendation["status"] == "karar_verilemedi": print("- Karar verilemedi: %s" % recommendation["reason"])


def main() -> None:
    articles, embeddings = load_inputs()
    subject_sets, display = build_subjects(articles)
    output = project_root() / "research" / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    runs: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    warnings: List[str] = []
    run_number = 0
    for method in METHODS:
        for seed in SEEDS:
            run_number += 1
            print("[%d/25] method=%s seed=%d" % (run_number, method["name"], seed), flush=True)
            try:
                row, prediction_rows = run_one(method, seed, articles, embeddings, subject_sets, display)
                runs.append(row); predictions.extend(prediction_rows)
                print("cluster=%d noise=%d top1=%.2f%% top2=%.2f%%" % (
                    row["train_cluster_count"], row["train_noise_count"],
                    100 * row["holdout_top1_metadata_consistency"], 100 * row["holdout_top2_metadata_consistency"]), flush=True)
                if row["warning"]: warnings.append("%s seed=%d: %s" % (method["name"], seed, row["warning"]))
            except (AssertionError, ValueError) as error:
                # Veri hizalama/sızıntı ve temel veri doğrulama hataları benchmarkı hemen durdurur.
                if isinstance(error, AssertionError) or "split" in str(error).lower() or "article_id" in str(error).lower() or "embedding" in str(error).lower():
                    raise
                runs.append(failed_row(method, seed, error)); warnings.append("%s seed=%d: %s" % (method["name"], seed, error))
                traceback.print_exc()
            except Exception as error:
                runs.append(failed_row(method, seed, error)); warnings.append("%s seed=%d: %s" % (method["name"], seed, error))
                traceback.print_exc()
    summaries = summarize(runs)
    recommendation = choose_recommendation(summaries)
    write_csv(output / "day28_stability_runs.csv", RUN_FIELDS, runs)
    write_csv(output / "day28_stability_method_summary.csv", SUMMARY_FIELDS, summaries)
    write_csv(output / "day28_holdout_predictions.csv", PREDICTION_FIELDS, predictions)
    settings = {"article_count": ARTICLE_COUNT, "train_count": TRAIN_COUNT, "test_count": TEST_COUNT,
                "split": "publication_year stratified", "embedding": "normalized TR-MTEB 768D",
                "silhouette_space": "original normalized 768D, cosine, noise excluded",
                "leakage_policy": "test subjects only after prediction", "methods": METHODS}
    payload = {"experiment_settings": settings, "seeds": SEEDS, "methods": [value["name"] for value in METHODS],
               "method_summaries": summaries, "warnings": warnings, "recommendation": recommendation}
    with (output / "day28_stability_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
    create_plots(runs, summaries, output)
    report = create_report(summaries, runs, recommendation)
    (output / "day28_stability_recommendation.md").write_text(report, encoding="utf-8")
    print_summary(summaries, recommendation)


if __name__ == "__main__":
    main()
