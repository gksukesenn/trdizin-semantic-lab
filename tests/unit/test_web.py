from trdizin_topic_pipeline.web.schemas import SearchRequest


def test_search_request_preserves_endpoint_defaults_and_converts_filters():
    request = SearchRequest.from_dict({"query": "eğitim", "year_from": "2020", "limit": "5"})
    assert request.mode == "semantic"
    assert request.limit == 5
    assert request.year_from == 2020
    assert request.year_to is None
