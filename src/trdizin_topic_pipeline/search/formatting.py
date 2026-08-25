"""Transforms raw Qdrant/RRF rows into the stable web response shape."""

from typing import Any, Dict, List


def _result_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title_tr": payload.get("title_tr", ""),
        "abstract_tr": payload.get("abstract_tr", ""),
        "publication_year": payload.get("publication_year"),
        "databases": payload.get("databases", []),
        "subjects": payload.get("subjects", []),
        "assignment_method": payload.get("assignment_method"),
        "primary_cluster": payload.get("primary_cluster"),
        "primary_topic": payload.get("primary_topic"),
        "secondary_cluster": payload.get("secondary_cluster"),
        "secondary_topic": payload.get("secondary_topic"),
        "primary_similarity": payload.get("primary_similarity"),
        "secondary_similarity": payload.get("secondary_similarity"),
        "similarity_margin": payload.get("similarity_margin"),
    }


def format_semantic_points(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for rank, point in enumerate(points, start=1):
        payload = point.get("payload", {})
        score = float(point.get("score", 0.0))
        results.append({
            "rank": rank, "article_id": payload.get("article_id"), "score": score,
            **_result_fields(payload),
            "search_details": {"mode": "semantic", "abstract_rank": rank, "abstract_score": score},
        })
    return results


def format_hybrid_results(
    fused: List[Dict[str, Any]], payload_by_id: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    results = []
    for final_rank, row in enumerate(fused, start=1):
        payload = payload_by_id.get(str(row.get("qdrant_id")), row.get("payload", {}))
        ranks, scores = row.get("ranks", {}), row.get("scores", {})
        rrf_score = float(row.get("rrf_score", 0.0))
        results.append({
            "rank": final_rank,
            "article_id": payload.get("article_id", row.get("article_id")),
            "score": rrf_score,
            **_result_fields(payload),
            "search_details": {
                "mode": "hybrid", "rrf_score": rrf_score,
                "abstract_rank": ranks.get("abstract"), "title_rank": ranks.get("title"),
                "bm25_rank": ranks.get("bm25"), "abstract_score": scores.get("abstract"),
                "title_score": scores.get("title"), "bm25_score": scores.get("bm25"),
                "matched_sources": row.get("matched_sources", []),
            },
        })
    return results
