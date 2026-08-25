import pytest

from trdizin_topic_pipeline.search.hybrid_search import multi_source_rrf, reciprocal_rank_fusion


def point(article_id, score):
    return {"id": article_id, "score": score, "payload": {"article_id": article_id}}


def test_rrf_rewards_article_present_in_both_rankings():
    result = reciprocal_rank_fusion([point("a", .9), point("b", .8)], [point("b", .95), point("c", .7)])
    assert result[0]["article_id"] == "b"
    assert result[0]["abstract_rank"] == 2
    assert result[0]["title_rank"] == 1


def test_multi_source_rrf_validates_parameters():
    with pytest.raises(ValueError):
        multi_source_rrf({}, rrf_k=0)
