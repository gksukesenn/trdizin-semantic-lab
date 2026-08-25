import numpy as np
import pytest

from trdizin_topic_pipeline.indexing.helpers import optional_float, optional_int, point_id, subject_names
from trdizin_topic_pipeline.indexing.payloads import build_abstract_payload, build_bm25_text, build_title_payload
from trdizin_topic_pipeline.indexing.validation import validate_collection, validate_vector_inputs


ARTICLE = {
    "article_id": "42", "title_tr": " Başlık ", "abstract_tr": "Özet",
    "publication_year": "2020", "keywords_tr": ["anahtar", " kelime "],
    "databases": ["SCIENCE"],
    "subjects": [{"fullName": "Eğitim"}, {"name": "Eğitim"}, "ignored"],
}
ASSIGNMENT = {"assignment_method": "direct", "primary_cluster": "3", "primary_similarity": "0.8"}


def test_point_id_is_numeric_or_stable_uuid():
    assert point_id("42") == 42
    assert point_id("article-x") == point_id("article-x")


def test_subject_and_optional_scalar_normalization():
    assert subject_names(ARTICLE) == ["Eğitim"]
    assert optional_int("") is None
    assert optional_int("3") == 3
    assert optional_float("bad") is None


def test_vector_payloads_keep_collection_specific_abstract_field():
    abstract = build_abstract_payload(0, ARTICLE, ASSIGNMENT)
    title = build_title_payload(0, ARTICLE, ASSIGNMENT)
    assert abstract["abstract_tr"] == "Özet"
    assert "abstract_tr" not in title
    assert abstract["subjects"] == ["Eğitim"]


def test_bm25_text_remains_title_plus_keywords_only():
    assert build_bm25_text(ARTICLE) == "Başlık anahtar kelime"
    assert "Özet" not in build_bm25_text(ARTICLE)


def test_collection_validation_checks_size_and_distance():
    info = {"result": {"config": {"params": {"vectors": {"size": 768, "distance": "Cosine"}}}}}
    validate_collection(info, 768, "cosine")
    with pytest.raises(ValueError, match="size"):
        validate_collection(info, 384, "cosine")


def test_common_vector_validation_checks_alignment_and_hash(tmp_path):
    quality = tmp_path / "quality.json"
    metadata = tmp_path / "metadata.json"
    quality.write_text('{"dataset_sha256":"same"}', encoding="utf-8")
    metadata.write_text('{"dataset_sha256":"same"}', encoding="utf-8")
    articles = [{"article_id": "42"}]
    assignments = [{"article_id": "42", "row_index": "0"}]
    vectors = np.zeros((1, 768), dtype=np.float32)
    assert validate_vector_inputs(
        articles, assignments, vectors, 1,
        quality_path=quality, metadata_path=metadata,
    ) == "same"
