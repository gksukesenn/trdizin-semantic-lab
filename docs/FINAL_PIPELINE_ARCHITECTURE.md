# Nihai Pipeline Mimarisi

```text
TR Dizin API
  → JSONL dataset
  → validation
  → TR-MTEB embedding (abstract_tr)
  → UMAP 10D
  → HDBSCAN Leaf
  → direct cluster / normalize centroid fallback
  → primary + secondary topic
  → ayrı UMAP 2D visualization
  → final report
```

`configs/final_50k.json` tek ayar kaynağıdır. `scripts/01...05` sıralı kullanıcı girişleridir. `src/trdizin_topic_pipeline/config.py` config doğrular; `io_utils.py` atomic dosya işlemlerini; `api_client.py` kanıtlanmış TR Dizin isteğini; `dataset.py` extraction ve deduplication'ı; `validation.py` kalite ölçümlerini; `embeddings.py` cihaz doğrulamalı resume embeddingini; `clustering.py` UMAP/HDBSCAN/fallback akışını; `evaluation.py` ölçeklenebilir metrikleri; `reporting.py` gerçek çıktı tabanlı raporu yönetir.

`data/state` collector checkpoint'ini tutar. `outputs/final_50k/embeddings`, `clustering`, `figures`, `reports`, `logs` ve `search` yeniden üretilebilir çıktıları ayırır. Day01–Day31 dosyaları ve çıktıları araştırma günlüğü olarak değişmeden kalır.

Docker bu tek makine, mevcut `.venv` ve doğrulanmış CUDA ortamında tekrar üretilebilirlik için gerekli değildir; ek operasyon yükü getirir. Qdrant batch konu keşfinin parçası değildir. Gelecekte interaktif semantic search/API gerekirse opsiyonel bir vektör indeks katmanı olabilir.

Clustering girdisi UMAP 10D'dir. UMAP 2D yalnız görselleştirme içindir; bilgi kaybeder ve cluster kararı üretmez. Subject/keyword metadata embedding veya clustering girdisi değildir; yalnız adlandırma ve keşifsel tutarlılık değerlendirmesinde kullanılır.
