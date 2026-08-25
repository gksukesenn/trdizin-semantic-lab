"""Abstract ve başlık arama sonuçlarını RRF ile birleştirir."""

from typing import Any, Dict, List


def point_key(point: Dict[str, Any]) -> str:
    """İki collection arasındaki ortak makale anahtarını döndürür."""

    payload = point.get("payload", {})

    article_id = str(
        payload.get("article_id", "")
    ).strip()

    if article_id:
        return article_id

    return str(point.get("id", "")).strip()


def reciprocal_rank_fusion(
    abstract_points: List[Dict[str, Any]],
    title_points: List[Dict[str, Any]],
    rrf_k: int = 60,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Abstract ve başlık sıralamalarını Reciprocal Rank Fusion ile birleştirir.

    Her sonuç için:

        RRF = 1 / (k + abstract_rank)
            + 1 / (k + title_rank)

    Bir sonuç yalnızca tek listede bulunuyorsa yalnız o listenin katkısını alır.
    """

    if rrf_k < 1:
        raise ValueError("rrf_k en az 1 olmalıdır.")

    if limit < 1:
        raise ValueError("limit en az 1 olmalıdır.")

    fused: Dict[str, Dict[str, Any]] = {}

    def add_points(
        points: List[Dict[str, Any]],
        source: str,
    ) -> None:
        for rank, point in enumerate(points, start=1):
            key = point_key(point)

            if not key:
                continue

            payload = point.get("payload", {})

            if key not in fused:
                fused[key] = {
                    "article_id": key,
                    "qdrant_id": point.get("id"),
                    "payload": payload,
                    "rrf_score": 0.0,
                    "abstract_rank": None,
                    "title_rank": None,
                    "abstract_score": None,
                    "title_score": None,
                }

            record = fused[key]

            record["rrf_score"] += (
                1.0 / float(rrf_k + rank)
            )

            if source == "abstract":
                record["abstract_rank"] = rank
                record["abstract_score"] = float(
                    point.get("score", 0.0)
                )

                # Abstract collection daha zengin payload taşıyor.
                record["payload"] = payload

            elif source == "title":
                record["title_rank"] = rank
                record["title_score"] = float(
                    point.get("score", 0.0)
                )

                if not record.get("payload"):
                    record["payload"] = payload

            else:
                raise ValueError(
                    "Bilinmeyen RRF kaynağı: %s" % source
                )

    add_points(
        abstract_points,
        source="abstract",
    )

    add_points(
        title_points,
        source="title",
    )

    results = list(fused.values())

    def minimum_rank(row: Dict[str, Any]) -> int:
        ranks = [
            rank
            for rank in (
                row.get("abstract_rank"),
                row.get("title_rank"),
            )
            if rank is not None
        ]

        return min(ranks) if ranks else 10**9

    def maximum_semantic_score(
        row: Dict[str, Any],
    ) -> float:
        scores = [
            float(score)
            for score in (
                row.get("abstract_score"),
                row.get("title_score"),
            )
            if score is not None
        ]

        return max(scores) if scores else 0.0

    results.sort(
        key=lambda row: (
            -float(row["rrf_score"]),
            minimum_rank(row),
            -maximum_semantic_score(row),
            str(row["article_id"]),
        )
    )
    return results[:limit]

def multi_source_rrf(
    ranked_sources: Dict[str, List[Dict[str, Any]]],
    rrf_k: int = 60,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Birden fazla sıralı sonuç listesini
    Reciprocal Rank Fusion ile birleştirir.
    """

    if rrf_k < 1:
        raise ValueError("rrf_k en az 1 olmalıdır.")

    if limit < 1:
        raise ValueError("limit en az 1 olmalıdır.")

    fused: Dict[str, Dict[str, Any]] = {}

    for source_name, points in ranked_sources.items():
        for rank, point in enumerate(points, start=1):
            key = point_key(point)

            if not key:
                continue

            payload = point.get("payload", {})

            if key not in fused:
                fused[key] = {
                    "article_id": key,
                    "qdrant_id": point.get("id"),
                    "payload": payload,
                    "rrf_score": 0.0,
                    "ranks": {},
                    "scores": {},
                    "matched_sources": [],
                }

            record = fused[key]

            record["rrf_score"] += (
                1.0 / float(rrf_k + rank)
            )

            record["ranks"][source_name] = rank
            record["scores"][source_name] = float(
                point.get("score", 0.0)
            )

            if source_name not in record["matched_sources"]:
                record["matched_sources"].append(
                    source_name
                )

            # Abstract collection daha zengin payload taşıyor.
            if source_name == "abstract":
                record["payload"] = payload
            elif not record.get("payload"):
                record["payload"] = payload

    def best_rank(row: Dict[str, Any]) -> int:
        ranks = list(
            row.get("ranks", {}).values()
        )

        if not ranks:
            return 10**9

        return min(ranks)

    results = list(fused.values())

    results.sort(
        key=lambda row: (
            -float(row["rrf_score"]),
            -len(row.get("matched_sources", [])),
            best_rank(row),
            str(row["article_id"]),
        )
    )

    return results[:limit]