import os

import pytest

from trdizin_topic_pipeline.search.qdrant_store import QdrantRestStore


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("TRDIZIN_QDRANT_INTEGRATION"), reason="real Qdrant opt-in değil")
def test_real_qdrant_collection_endpoint():
    store = QdrantRestStore(os.getenv("TRDIZIN_QDRANT_URL", "http://127.0.0.1:6335"))
    assert isinstance(store.collection_exists("trdizin_articles_50000"), bool)
