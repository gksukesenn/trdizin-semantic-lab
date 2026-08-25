"""Metadata relevance and standard retrieval metrics."""

import math
import unicodedata
from typing import Any, Dict, List, Sequence


def normalize_text(value: Any) -> str:
    text = str(value or "").casefold()
    text = (text.replace("ı", "i").replace("ş", "s").replace("ğ", "g")
            .replace("ü", "u").replace("ö", "o").replace("ç", "c"))
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(character for character in decomposed if not unicodedata.combining(character))
    return " ".join(without_marks.split())


def metadata_text(payload: Dict[str, Any]) -> str:
    values: List[str] = []
    subjects = payload.get("subjects", [])
    if isinstance(subjects, list):
        values.extend(str(value) for value in subjects)
    values.append(str(payload.get("primary_topic", "")))
    values.append(str(payload.get("secondary_topic", "")))
    return normalize_text(" ".join(values))


def is_metadata_relevant(payload: Dict[str, Any], relevance_groups: Sequence[Sequence[str]]) -> bool:
    """OR between groups, AND between normalized fragments in one group."""
    searchable = metadata_text(payload)
    for group in relevance_groups:
        fragments = [normalize_text(fragment) for fragment in group if str(fragment).strip()]
        if fragments and all(fragment in searchable for fragment in fragments):
            return True
    return False


def precision_at_k(relevance: List[int], k: int) -> float:
    selected = relevance[:k]
    return sum(selected) / float(len(selected)) if selected else 0.0


def reciprocal_rank(relevance: List[int]) -> float:
    for index, value in enumerate(relevance, start=1):
        if value:
            return 1.0 / float(index)
    return 0.0


def dcg_at_k(relevance: List[int], k: int) -> float:
    return sum(1.0 / math.log2(index + 1) for index, value in enumerate(relevance[:k], start=1) if value)


def ndcg_at_k(relevance: List[int], k: int) -> float:
    actual = dcg_at_k(relevance, k)
    ideal = dcg_at_k(sorted(relevance[:k], reverse=True), k)
    return actual / ideal if ideal else 0.0
