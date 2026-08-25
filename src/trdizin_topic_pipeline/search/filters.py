"""Qdrant filter construction shared by web and search CLIs."""

from typing import Any, Dict, List, Optional


def build_filter(
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    database: Optional[str] = None,
    primary_cluster: Optional[int] = None,
    assignment_method: Optional[str] = None,
    primary_topic: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    must: List[Dict[str, Any]] = []
    if year_from is not None or year_to is not None:
        value: Dict[str, int] = {}
        if year_from is not None:
            value["gte"] = year_from
        if year_to is not None:
            value["lte"] = year_to
        must.append({"key": "publication_year", "range": value})
    if database:
        must.append({"key": "databases", "match": {"value": database.upper()}})
    if primary_cluster is not None:
        must.append({"key": "primary_cluster", "match": {"value": primary_cluster}})
    if assignment_method:
        must.append({"key": "assignment_method", "match": {"value": assignment_method}})
    if primary_topic:
        must.append({"key": "primary_topic", "match": {"value": primary_topic}})
    return {"must": must} if must else None
