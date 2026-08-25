import numpy as np
import pytest

from trdizin_topic_pipeline.search.filters import build_filter
from trdizin_topic_pipeline.search.formatting import format_hybrid_results, format_semantic_points
from trdizin_topic_pipeline.search.query_encoder import QueryEncoder, select_device


class FakeCuda:
    @staticmethod
    def is_available():
        return False


class FakeTorch:
    cuda = FakeCuda()


class FakeModel:
    def __init__(self, values):
        self.values = values

    def encode(self, *args, **kwargs):
        return self.values


def test_filter_builder_combines_all_supported_conditions():
    value = build_filter(2019, 2022, "science", 7, "direct", "Eğitim")
    assert value == {"must": [
        {"key": "publication_year", "range": {"gte": 2019, "lte": 2022}},
        {"key": "databases", "match": {"value": "SCIENCE"}},
        {"key": "primary_cluster", "match": {"value": 7}},
        {"key": "assignment_method", "match": {"value": "direct"}},
        {"key": "primary_topic", "match": {"value": "Eğitim"}},
    ]}


def test_query_encoder_uses_mock_model_without_download():
    values = np.ones((1, 768), dtype=np.float32) / np.sqrt(768)
    encoder = QueryEncoder("unused", allow_cpu=True, model=FakeModel(values), torch_module=FakeTorch())
    vector, elapsed = encoder.encode("sorgu")
    assert encoder.device == "cpu"
    assert vector.shape == (768,)
    assert elapsed >= 0


def test_device_selection_requires_explicit_cpu_fallback():
    with pytest.raises(RuntimeError, match="allow-cpu"):
        select_device(False, FakeTorch())


def test_semantic_formatter_keeps_response_contract():
    result = format_semantic_points([{"score": .75, "payload": {"article_id": "1", "title_tr": "T"}}])[0]
    assert result["rank"] == 1
    assert result["score"] == .75
    assert result["search_details"] == {"mode": "semantic", "abstract_rank": 1, "abstract_score": .75}


def test_hybrid_formatter_prefers_retrieved_abstract_payload():
    fused = [{"qdrant_id": 4, "article_id": "old", "rrf_score": .1, "ranks": {"bm25": 1}, "scores": {"bm25": 3.0}, "matched_sources": ["bm25"], "payload": {}}]
    result = format_hybrid_results(fused, {"4": {"article_id": "new", "abstract_tr": "A"}})[0]
    assert result["article_id"] == "new"
    assert result["search_details"]["bm25_rank"] == 1
