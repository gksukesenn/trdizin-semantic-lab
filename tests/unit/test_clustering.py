import numpy as np
import pytest

from trdizin_topic_pipeline.topics.clustering import normalized_centroid


def test_normalized_centroid_preserves_unit_norm():
    centroid = normalized_centroid(np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    assert centroid.dtype == np.float32
    assert float(np.linalg.norm(centroid)) == pytest.approx(1.0)
