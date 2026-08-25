"""Qdrant payload construction without changing collection-specific fields."""

from typing import Any, Dict

from .helpers import optional_float, optional_int, subject_names


def build_vector_payload(
    row_index: int, article: Dict[str, Any], assignment: Dict[str, str],
    include_abstract: bool,
) -> Dict[str, Any]:
    payload = {
        "row_index": row_index,
        "article_id": str(article["article_id"]),
        "title_tr": str(article.get("title_tr", "")),
        "publication_year": optional_int(article.get("publication_year")),
        "keywords_tr": [str(value) for value in article.get("keywords_tr", [])],
        "databases": [str(value) for value in article.get("databases", [])],
        "subjects": subject_names(article),
        "raw_hdbscan_cluster": optional_int(assignment.get("raw_hdbscan_cluster")),
        "assignment_method": str(assignment.get("assignment_method", "")),
        "primary_cluster": optional_int(assignment.get("primary_cluster")),
        "primary_topic": str(assignment.get("primary_topic", "")),
        "primary_similarity": optional_float(assignment.get("primary_similarity")),
        "secondary_cluster": optional_int(assignment.get("secondary_cluster")),
        "secondary_topic": str(assignment.get("secondary_topic", "")),
        "secondary_similarity": optional_float(assignment.get("secondary_similarity")),
        "similarity_margin": optional_float(assignment.get("similarity_margin")),
    }
    if include_abstract:
        payload["abstract_tr"] = str(article.get("abstract_tr", ""))
    return {key: value for key, value in payload.items() if value is not None}


def build_abstract_payload(row_index: int, article: Dict[str, Any], assignment: Dict[str, str]) -> Dict[str, Any]:
    return build_vector_payload(row_index, article, assignment, include_abstract=True)


def build_title_payload(row_index: int, article: Dict[str, Any], assignment: Dict[str, str]) -> Dict[str, Any]:
    return build_vector_payload(row_index, article, assignment, include_abstract=False)


def build_bm25_text(article: Dict[str, Any]) -> str:
    title = " ".join(str(article.get("title_tr", "")).split())
    keywords = [" ".join(str(value).split()) for value in article.get("keywords_tr", []) if str(value).strip()]
    parts = [title]
    if keywords:
        parts.append(" ".join(keywords))
    return " ".join(part for part in parts if part)
