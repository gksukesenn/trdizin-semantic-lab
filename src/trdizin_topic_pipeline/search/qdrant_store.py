"""Qdrant REST API için küçük ve bağımlılığı az istemci."""

from typing import Any, Dict, List, Optional, Tuple

import requests


class QdrantRequestError(RuntimeError):
    """Qdrant API isteği başarısız olduğunda fırlatılır."""


class QdrantRestStore:
    """Qdrant REST API işlemlerini kapsüller."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: int = 180,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def close(self) -> None:
        self.session.close()

    def _request(
        self,
        method: str,
        path: str,
        expected_statuses: Tuple[int, ...] = (200,),
        json_body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = self.base_url + path

        response = self.session.request(
            method=method,
            url=url,
            json=json_body,
            params=params,
            timeout=self.timeout_seconds,
        )

        if response.status_code not in expected_statuses:
            raise QdrantRequestError(
                "Qdrant isteği başarısız.\n"
                "Method : %s\n"
                "URL    : %s\n"
                "Status : %s\n"
                "Yanıt  : %s"
                % (
                    method,
                    url,
                    response.status_code,
                    response.text[:1000],
                )
            )

        if not response.content:
            return {}

        try:
            data = response.json()
        except ValueError as error:
            raise QdrantRequestError(
                "Qdrant JSON olmayan yanıt döndürdü: %s"
                % response.text[:1000]
            ) from error

        if (
            isinstance(data, dict)
            and data.get("status") not in (None, "ok")
        ):
            raise QdrantRequestError(
                "Qdrant işlem durumu başarısız: %r"
                % data
            )

        return data

    def collection_exists(
        self,
        collection_name: str,
    ) -> bool:
        response = self.session.get(
            self.base_url
            + "/collections/"
            + collection_name,
            timeout=self.timeout_seconds,
        )

        if response.status_code == 404:
            return False

        if response.status_code != 200:
            raise QdrantRequestError(
                "Collection kontrolü başarısız: %s"
                % response.text[:1000]
            )

        return True

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str,
    ) -> None:
        self._request(
            method="PUT",
            path="/collections/" + collection_name,
            json_body={
                "vectors": {
                    "size": vector_size,
                    "distance": distance,
                },
                "on_disk_payload": True,
            },
        )

    def delete_collection(
        self,
        collection_name: str,
    ) -> None:
        self._request(
            method="DELETE",
            path="/collections/" + collection_name,
        )

    def collection_info(
        self,
        collection_name: str,
    ) -> Dict[str, Any]:
        return self._request(
            method="GET",
            path="/collections/" + collection_name,
        )

    def upsert_points(
        self,
        collection_name: str,
        points: List[Dict[str, Any]],
    ) -> None:
        self._request(
            method="PUT",
            path=(
                "/collections/"
                + collection_name
                + "/points"
            ),
            params={"wait": "true"},
            json_body={"points": points},
        )

    def exact_count(
        self,
        collection_name: str,
    ) -> int:
        data = self._request(
            method="POST",
            path=(
                "/collections/"
                + collection_name
                + "/points/count"
            ),
            json_body={"exact": True},
        )

        return int(
            data.get("result", {}).get("count", 0)
        )

    def query_points(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        query_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Bir sorgu vektörüne en yakın Qdrant pointlerini getirir."""

        body: Dict[str, Any] = {
            "query": query_vector,
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        }

        if query_filter:
            body["filter"] = query_filter

        data = self._request(
            method="POST",
            path=(
                "/collections/"
                + collection_name
                + "/points/query"
            ),
            json_body=body,
        )

        points = (
            data.get("result", {})
            .get("points", [])
        )

        if not isinstance(points, list):
            raise QdrantRequestError(
                "Qdrant query yanıtında points listesi bulunamadı."
            )

        return points


    def retrieve_points(
        self,
        collection_name: str,
        point_ids: List[Any],
        with_payload: bool = True,
        with_vector: bool = False,
    ) -> List[Dict[str, Any]]:
        """Belirli Qdrant point ID'lerini toplu olarak getirir."""

        if not point_ids:
            return []

        data = self._request(
            method="POST",
            path=(
                "/collections/"
                + collection_name
                + "/points"
            ),
            json_body={
                "ids": point_ids,
                "with_payload": with_payload,
                "with_vector": with_vector,
            },
        )

        points = data.get("result", [])

        if not isinstance(points, list):
            raise QdrantRequestError(
                "Point retrieval yanıtı liste değil."
            )

        return points

    def query_bm25_points(
        self,
        collection_name: str,
        query_text: str,
        limit: int = 50,
        query_filter: Optional[Dict[str, Any]] = None,
        vector_name: str = "text_bm25",
    ) -> List[Dict[str, Any]]:
        """Qdrant BM25 sparse collection üzerinde metin sorgusu yapar."""

        body: Dict[str, Any] = {
            "query": {
                "text": query_text,
                "model": "qdrant/bm25",
                "options": {
                    "language": "none",
                    "tokenizer": "multilingual",
                },
            },
            "using": vector_name,
            "limit": limit,
            "with_payload": True,
            "with_vector": False,
        }

        if query_filter:
            body["filter"] = query_filter

        data = self._request(
            method="POST",
            path=(
                "/collections/"
                + collection_name
                + "/points/query"
            ),
            json_body=body,
        )

        points = (
            data.get("result", {})
            .get("points", [])
        )

        if not isinstance(points, list):
            raise QdrantRequestError(
                "BM25 sorgu yanıtında points listesi bulunamadı."
            )

        return points
    def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema: str,
    ) -> None:
        url = (
            self.base_url
            + "/collections/"
            + collection_name
            + "/index"
        )

        response = self.session.put(
            url,
            params={"wait": "true"},
            json={
                "field_name": field_name,
                "field_schema": field_schema,
            },
            timeout=self.timeout_seconds,
        )

        if response.status_code == 200:
            return

        # Index daha önce oluşturulmuşsa tekrar çalıştırmayı bozma.
        if response.status_code == 400:
            lowered = response.text.casefold()

            if (
                "already exists" in lowered
                or "already indexed" in lowered
            ):
                return

        raise QdrantRequestError(
            "Payload index oluşturulamadı.\n"
            "Alan   : %s\n"
            "Şema   : %s\n"
            "Status : %s\n"
            "Yanıt  : %s"
            % (
                field_name,
                field_schema,
                response.status_code,
                response.text[:1000],
            )
        )