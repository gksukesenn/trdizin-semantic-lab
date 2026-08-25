"""Cluster metadata tutarlılığı ve ölçeklenebilir kalite metrikleri."""

from collections import Counter
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from sklearn.metrics import silhouette_score

from ..data.validation import subject_names


def subject_sets(rows: List[Dict[str, Any]]) -> List[Set[str]]:
    return [set(subject_names(row)) for row in rows]


def weighted_subject_purity(labels: np.ndarray, subjects: List[Set[str]]) -> Tuple[float, int]:
    numerator = denominator = 0
    for cluster_id in sorted(set(int(value) for value in labels if int(value) >= 0)):
        indices = np.where(labels == cluster_id)[0]
        labeled = [subjects[int(index)] for index in indices if subjects[int(index)]]
        counts = Counter(value for values in labeled for value in values)
        if counts:
            numerator += counts.most_common(1)[0][1]; denominator += len(labeled)
    return (numerator / float(denominator) if denominator else 0.0), denominator


def sampled_shared_subject_rate(labels: np.ndarray, subjects: List[Set[str]], sample_size: int, seed: int) -> float:
    rng = np.random.RandomState(seed); matches = tested = 0
    clusters = [value for value in sorted(set(int(v) for v in labels if int(v) >= 0)) if np.sum(labels == value) >= 2]
    if not clusters: return 0.0
    for _ in range(sample_size):
        cluster = clusters[int(rng.randint(len(clusters)))]; indices = np.where(labels == cluster)[0]
        pair = rng.choice(indices, size=2, replace=False); left, right = subjects[int(pair[0])], subjects[int(pair[1])]
        if left and right: tested += 1; matches += int(bool(left & right))
    return matches / float(tested) if tested else 0.0


def cosine_silhouette(embeddings: np.ndarray, labels: np.ndarray, sample_size: int, seed: int) -> Any:
    mask = labels >= 0; clean_labels = labels[mask]
    if mask.sum() < 2 or len(set(int(value) for value in clean_labels)) < 2: return None
    return float(silhouette_score(embeddings[mask], clean_labels, metric="cosine",
                                  sample_size=min(sample_size, int(mask.sum())), random_state=seed))
