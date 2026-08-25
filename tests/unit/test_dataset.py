import pytest

from trdizin_topic_pipeline.data.dataset import extract_article, validate_core


def test_extract_article_normalizes_trdizin_hit():
    hit = {"_source": {"id": "42", "language": "TUR", "docType": "PAPER", "abstracts": [{"language": "TUR", "title": "  Başlık ", "abstract": " Geçerli   özet ", "keywords": [" anahtar "]}], "subjects": ['{"name": "Eğitim"}']}}
    row = extract_article(hit)
    assert row is not None
    assert row["article_id"] == "42"
    assert row["abstract_tr"] == "Geçerli özet"
    assert row["keywords_tr"] == ["anahtar"]


def test_validate_core_rejects_duplicate_ids():
    rows = [{"article_id": "1", "abstract_tr": "bir", "title_tr": "a"}, {"article_id": "1", "abstract_tr": "iki", "title_tr": "b"}]
    with pytest.raises(ValueError, match="article_id"):
        validate_core(rows, expected=2, pilot_ids=set(), validation_ids=set())
