#!/usr/bin/env python3
"""Hazır embedding üzerinde CPU tabanlı topic discovery ve baseline çalıştırır."""
import argparse, json, os, sys, time
from pathlib import Path
from typing import Any, Dict, List
ROOT = Path(__file__).resolve().parents[4]; os.environ.setdefault("MPLCONFIGDIR", "/tmp/trdizin-matplotlib")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from trdizin_topic_pipeline.topics.clustering import assign_primary_secondary, choose_configuration, cluster_dictionary, evaluate_configuration, fit_hdbscan, reduce_umap
from trdizin_topic_pipeline.config import ensure_output_directories, load_config, resolve_path
from trdizin_topic_pipeline.topics.evaluation import subject_sets, weighted_subject_purity
from trdizin_topic_pipeline.utils.io import atomic_csv, atomic_json, banner, file_sha256, read_jsonl

def save_plot(path: Path) -> None: path.parent.mkdir(parents=True, exist_ok=True); plt.tight_layout(); plt.savefig(str(path), dpi=160); plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="configs/final_50k.json"); parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(); config = load_config(Path(args.config)); ensure_output_directories(config); output = resolve_path(config, "output_root")
    dataset = resolve_path(config, "dataset"); embedding_path = resolve_path(config, "embedding"); rows = read_jsonl(dataset)
    if args.smoke_test:
        smoke = output.parent / "smoke_test"; dataset = resolve_path(config, "validation_dataset"); rows = read_jsonl(dataset)[:200]
        source = ROOT / "research/outputs/day29_embeddings/tr_mteb_validation_5000.npy"; embeddings = np.load(str(source), mmap_mode="r")[:200]; output = smoke
    else:
        if len(rows) != int(config["target_article_count"]): raise ValueError("Topic discovery tam veri seti gerektirir.")
        embeddings = np.load(str(embedding_path), mmap_mode="r")
        with resolve_path(config, "embedding_metadata").open("r", encoding="utf-8") as handle: metadata = json.load(handle)
        if metadata.get("dataset_sha256") != file_sha256(dataset): raise ValueError("Dataset/embedding SHA-256 hizalaması başarısız.")
        if not metadata.get("row_alignment"): raise ValueError("Embedding satır hizalama metadata'sı eksik.")
    if embeddings.shape[0] != len(rows): raise ValueError("JSONL/embedding satır sayısı farklı.")
    banner("50.000 MAKALE TOPIC DISCOVERY")
    print("Embedding dosyası hazırdır. UMAP, HDBSCAN ve scikit-learn KMeans aşamaları CPU tabanlıdır; GPU kullanımı beklenmez.")
    seed = int(config["random_seed"]); reduced = reduce_umap(np.asarray(embeddings), config["umap"], seed, 10); subjects = subject_sets(rows); sweep = []; label_map = {}
    for size in config["hdbscan"]["min_cluster_size_candidates"]:
        for samples in config["hdbscan"]["min_samples_candidates"]:
            started = time.time(); labels = fit_hdbscan(reduced, int(size), int(samples)); runtime = time.time() - started
            result = evaluate_configuration(np.asarray(embeddings), reduced, labels, subjects, config, int(size), int(samples), runtime)
            sweep.append(result); label_map[(int(size), int(samples))] = labels
            print("mcs=%d ms=%d cluster=%d noise=%.2f%% süre=%.1f sn" % (size, samples, result["cluster_count"], result["noise_rate"] * 100, runtime))
    chosen, reason = choose_configuration(sweep); clustering = output / "clustering"; figures = output / "figures"
    fields = list(sweep[0].keys()); atomic_csv(clustering / "hdbscan_parameter_sweep.csv", sweep, fields)
    if chosen is None:
        atomic_json(clustering / "final_topic_pipeline_summary.json", {"selection_status": "karar_verilemedi", "selection_reason": reason, "parameter_results": sweep}); print(reason); return
    labels = label_map[(int(chosen["min_cluster_size"]), int(chosen["min_samples"]))]
    cluster_ids, centroids, dictionary = cluster_dictionary(rows, np.asarray(embeddings), labels, subjects); assignments, fallbacks = assign_primary_secondary(rows, np.asarray(embeddings), labels, cluster_ids, centroids, dictionary)
    clustering.mkdir(parents=True, exist_ok=True); np.save(str(clustering / "cluster_centroids.npy"), centroids)
    atomic_csv(clustering / "final_cluster_assignments.csv", assignments, list(assignments[0].keys())); atomic_csv(clustering / "final_cluster_dictionary.csv", dictionary, list(dictionary[0].keys()))
    fallback_fields = list(assignments[0].keys()); atomic_csv(clustering / "noise_fallback_assignments.csv", fallbacks, fallback_fields)
    kcfg = config["kmeans"]; klabels = KMeans(n_clusters=int(kcfg["n_clusters"]), n_init=int(kcfg["n_init"]), random_state=seed).fit_predict(embeddings)
    krows = [{"row_index": index, "article_id": rows[index]["article_id"], "kmeans_cluster": int(label)} for index, label in enumerate(klabels)]
    atomic_csv(clustering / "kmeans_baseline_assignments.csv", krows, list(krows[0].keys())); kpurity, klabeled = weighted_subject_purity(klabels, subjects)
    summary = {"selection_status": "seçildi", "selected_parameters": {"min_cluster_size": chosen["min_cluster_size"], "min_samples": chosen["min_samples"], "cluster_selection_method": "leaf"},
               "selection_reason": reason, "cluster_space": "UMAP 10D", "visualization_space": "ayrı UMAP 2D; clustering girdisi değildir", "cluster_count": len(cluster_ids),
               "noise_count": len(fallbacks), "noise_rate": len(fallbacks) / float(len(rows)), "direct_count": len(rows) - len(fallbacks), "dataset_row_count": len(rows),
               "kmeans": {"n_clusters": int(kcfg["n_clusters"]), "weighted_subject_purity": kpurity, "subject_labeled_count": klabeled}, "random_seed": seed, "parameter_results": sweep}
    atomic_json(clustering / "final_topic_pipeline_summary.json", summary)
    coords = reduce_umap(np.asarray(embeddings), config["umap"], seed, 2)
    plt.figure(figsize=(10, 7)); plt.scatter(coords[:, 0], coords[:, 1], c=[item["primary_cluster"] for item in assignments], s=3, cmap="tab20", alpha=.65); plt.title("UMAP 2D – nihai konular"); save_plot(figures / "umap_2d_clusters.png")
    plt.figure(figsize=(10, 7)); plt.scatter(coords[:, 0], coords[:, 1], c=[0 if item["assignment_method"] == "direct" else 1 for item in assignments], s=3, cmap="coolwarm", alpha=.65); plt.title("UMAP 2D – direct / centroid fallback"); save_plot(figures / "umap_2d_direct_fallback.png")
    plt.figure(figsize=(8, 5)); plt.hist([row["cluster_size"] for row in dictionary], bins=30); plt.title("Cluster boyutu dağılımı"); save_plot(figures / "cluster_size_distribution.png")
    plt.figure(figsize=(9, 5)); labels_x = ["%s/%s" % (row["min_cluster_size"], row["min_samples"]) for row in sweep]; plt.plot(labels_x, [row["noise_rate"] for row in sweep], marker="o", label="Noise oranı"); plt.plot(labels_x, [row["weighted_subject_purity"] for row in sweep], marker="s", label="Weighted subject purity"); plt.legend(); plt.title("Parametre karşılaştırması (ayrı metrikler)"); save_plot(figures / "parameter_comparison.png")
    print("Seçim: mcs=%s, min_samples=%s; %s" % (chosen["min_cluster_size"], chosen["min_samples"], reason))

if __name__ == "__main__": main()
