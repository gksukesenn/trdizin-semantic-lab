import numpy as np

from trdizin_topic_pipeline.topics.embeddings import duplicate_vector_count


def test_duplicate_vector_count_counts_repeated_rows():
    values = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    assert duplicate_vector_count(values) == 1
