"""UMAP10 + HDBSCAN Leaf, centroid fallback ve KMeans baseline."""

import time
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from .evaluation import cosine_silhouette, sampled_shared_subject_rate, weighted_subject_purity


def normalized_centroid(values: np.ndarray) -> np.ndarray:
    centroid = values.mean(axis=0); norm = float(np.linalg.norm(centroid))
    if norm == 0.0: raise ValueError("Sıfır normlu cluster centroidi.")
    return (centroid / norm).astype(np.float32)


def reduce_umap(embeddings: np.ndarray, settings: Dict[str, Any], seed: int, components: int) -> np.ndarray:
    import umap
    reducer = umap.UMAP(n_components=components, n_neighbors=int(settings["n_neighbors"]),
                        min_dist=float(settings["min_dist"]), metric=str(settings["metric"]),
                        random_state=seed, low_memory=True)
    return reducer.fit_transform(embeddings).astype(np.float32)


def fit_hdbscan(reduced: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
    import hdbscan
    return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                           cluster_selection_method="leaf", metric="euclidean",
                           gen_min_span_tree=False).fit_predict(reduced).astype(np.int32)


def evaluate_configuration(embeddings: np.ndarray, reduced: np.ndarray, labels: np.ndarray,
                           subjects: List[Set[str]], config: Dict[str, Any], min_cluster_size: int,
                           min_samples: int, runtime: float) -> Dict[str, Any]:
    cluster_ids = sorted(set(int(value) for value in labels if int(value) >= 0))
    sizes = [int(np.sum(labels == value)) for value in cluster_ids]
    noise = int(np.sum(labels < 0)); warnings = []
    if len(cluster_ids) < 5: warnings.append("cluster çökmesi olasılığı")
    if noise / float(len(labels)) > 0.60: warnings.append("noise oranı %60 üzerinde")
    purity, labeled = weighted_subject_purity(labels, subjects)
    evaluation = config["evaluation"]; seed = int(config["random_seed"])
    return {"min_cluster_size": min_cluster_size, "min_samples": min_samples,
            "cluster_selection_method": "leaf", "cluster_count": len(cluster_ids),
            "noise_count": noise, "noise_rate": noise / float(len(labels)),
            "min_observed_cluster_size": min(sizes) if sizes else 0,
            "median_cluster_size": float(np.median(sizes)) if sizes else 0.0,
            "max_cluster_size": max(sizes) if sizes else 0,
            "weighted_subject_purity": purity, "subject_labeled_count": labeled,
            "sampled_shared_subject_pair_rate": sampled_shared_subject_rate(labels, subjects, int(evaluation["pair_sample_size"]), seed),
            "cosine_silhouette": cosine_silhouette(embeddings, labels, int(evaluation["silhouette_sample_size"]), seed),
            "small_cluster_count": sum(size < int(evaluation["small_cluster_threshold"]) for size in sizes),
            "very_large_cluster_count": sum(size > len(labels) * float(evaluation["large_cluster_fraction"]) for size in sizes),
            "runtime_seconds": runtime, "warnings": " | ".join(warnings)}


def choose_configuration(results: List[Dict[str, Any]]) -> Tuple[Any, str]:
    viable = [row for row in results if row["cluster_count"] >= 5 and row["noise_rate"] <= 0.60 and row["very_large_cluster_count"] == 0]
    if not viable:
        return None, "Karar verilemedi: bütün adaylarda çökme, aşırı noise veya aşırı büyük cluster gözlendi."
    # Composite score yok: sırasıyla çökme/noise, küçük cluster, metadata tutarlılığı, silhouette.
    viable.sort(key=lambda row: (row["small_cluster_count"], -row["weighted_subject_purity"],
                                 -(row["cosine_silhouette"] if row["cosine_silhouette"] is not None else -999),
                                 row["noise_rate"], row["min_cluster_size"], row["min_samples"]))
    chosen = viable[0]
    reason = ("Çökme ve aşırı noise göstermeyen adaylar içinde önce küçük cluster sayısı, sonra weighted subject "
              "metadata tutarlılığı ve cosine silhouette karşılaştırıldı; composite score kullanılmadı.")
    return chosen, reason


def cluster_dictionary(rows: List[Dict[str, Any]], embeddings: np.ndarray, labels: np.ndarray,
                       subjects: List[Set[str]]) -> Tuple[List[int], np.ndarray, List[Dict[str, Any]]]:
    cluster_ids = sorted(set(int(value) for value in labels if int(value) >= 0)); centroids = []; dictionary = []
    for cluster_id in cluster_ids:
        indices = np.where(labels == cluster_id)[0]; centroid = normalized_centroid(embeddings[indices]); centroids.append(centroid)
        similarities = embeddings[indices] @ centroid; medoid_index = int(indices[int(np.argmax(similarities))])
        subject_counts = Counter(value for index in indices for value in subjects[int(index)])
        keyword_counts = Counter(str(value).strip() for index in indices for value in rows[int(index)].get("keywords_tr", []) if str(value).strip())
        top_subjects = [value for value, _ in subject_counts.most_common(3)]
        labeled = sum(bool(subjects[int(index)]) for index in indices)
        dictionary.append({"cluster_id": cluster_id, "temporary_topic_name": top_subjects[0] if top_subjects else "Cluster %d" % cluster_id,
                           "cluster_size": len(indices), "subject_purity": subject_counts.most_common(1)[0][1] / float(labeled) if labeled and subject_counts else 0.0,
                           "top_subjects": " | ".join(top_subjects),
                           "top_keywords": " | ".join(value for value, _ in keyword_counts.most_common(10)),
                           "medoid_article_id": rows[medoid_index]["article_id"], "medoid_title": rows[medoid_index].get("title_tr", ""),
                           "representative_titles": " | ".join(str(rows[int(index)].get("title_tr", "")) for index in indices[np.argsort(similarities)[::-1][:5]])})
    return cluster_ids, np.vstack(centroids).astype(np.float32), dictionary


def assign_primary_secondary(rows: List[Dict[str, Any]], embeddings: np.ndarray, labels: np.ndarray,
                             cluster_ids: List[int], centroids: np.ndarray,
                             dictionary: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    names = {int(row["cluster_id"]): row["temporary_topic_name"] for row in dictionary}; scores = embeddings @ centroids.T
    assignments = []; fallbacks = []
    for index, row in enumerate(rows):
        ranking = np.argsort(scores[index])[::-1]; raw = int(labels[index])
        primary = raw if raw >= 0 else cluster_ids[int(ranking[0])]
        secondary = next(cluster_ids[int(position)] for position in ranking if cluster_ids[int(position)] != primary)
        positions = {value: pos for pos, value in enumerate(cluster_ids)}
        primary_similarity = float(scores[index, positions[primary]]); secondary_similarity = float(scores[index, positions[secondary]])
        item = {"row_index": index, "article_id": row["article_id"], "raw_hdbscan_cluster": raw,
                "assignment_method": "direct" if raw >= 0 else "centroid_fallback",
                "primary_cluster": primary, "primary_topic": names[primary], "primary_similarity": primary_similarity,
                "secondary_cluster": secondary, "secondary_topic": names[secondary], "secondary_similarity": secondary_similarity,
                "similarity_margin": primary_similarity - secondary_similarity}
        assignments.append(item)
        if raw < 0: fallbacks.append(dict(item))
    return assignments, fallbacks
