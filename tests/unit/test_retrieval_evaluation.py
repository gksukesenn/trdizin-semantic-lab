import pytest

from trdizin_topic_pipeline.evaluation.retrieval import (
    dcg_at_k, is_metadata_relevant, ndcg_at_k, precision_at_k, reciprocal_rank,
)


def test_retrieval_metrics_keep_original_definitions():
    relevance = [0, 1, 1, 0]
    assert precision_at_k(relevance, 2) == .5
    assert reciprocal_rank(relevance) == .5
    assert dcg_at_k(relevance, 2) == pytest.approx(1 / 1.584962500721156)
    assert 0 < ndcg_at_k(relevance, 4) <= 1


def test_benchmark_relevance_uses_or_groups_and_and_fragments():
    payload = {"subjects": ["Eğitim Bilimleri"], "primary_topic": "Öğretmen Eğitimi"}
    assert is_metadata_relevant(payload, [["egitim", "ogretmen"], ["hukuk"]])
    assert not is_metadata_relevant(payload, [["egitim", "hukuk"]])
