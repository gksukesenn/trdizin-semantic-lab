from trdizin_topic_pipeline.search.qdrant_store import QdrantRestStore


class Response:
    status_code = 200
    content = b"json"
    text = ""

    def json(self):
        return {"status": "ok", "result": {"points": [{"id": 1, "payload": {"article_id": "a"}}]}}


class Session:
    def __init__(self):
        self.call = None

    def request(self, **kwargs):
        self.call = kwargs
        return Response()


def test_query_points_builds_qdrant_request_and_unwraps_points():
    store = QdrantRestStore("http://qdrant/")
    store.session = Session()
    points = store.query_points("articles", [0.1, 0.2], limit=3, query_filter={"must": []})
    assert points[0]["payload"]["article_id"] == "a"
    assert store.session.call["url"].endswith("/collections/articles/points/query")
    assert store.session.call["json"]["limit"] == 3
