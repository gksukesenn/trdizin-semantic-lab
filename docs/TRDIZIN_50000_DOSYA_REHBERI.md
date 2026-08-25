# TR Dizin 50K — Repository Dosya Rehberi

## Üst düzey harita

| Path | Ne işe yarar? | Değiştirme riski |
|---|---|---|
| `configs/` | Final ayarlar ve benchmark sorguları | Parametre değişirse çıktılar karşılaştırılamaz |
| `data/raw/` | API ara/ham verileri | Yeniden üretimi maliyetli |
| `data/processed/` | Pilot 1K, validation 5K, final 50K JSONL | Embedding sıra hizasını bozar |
| `data/state/` | Collection checkpoint/resume durumu | Toplamayı tekrar/eksik başlatabilir |
| `src/trdizin_topic_pipeline/` | Yeniden kullanılan pipeline modülleri | Tüm scriptleri etkileyebilir |
| `src/day*.py` | Tarihsel deneyler | Final operasyon hattıyla karıştırılmamalı |
| `scripts/` | Final 01–14 komutları | Bazıları veri/collection yazar |
| `outputs/final_50k/` | Final kalıcı kanıtlar | Raporun sayısal kaynağı |
| `infra/qdrant/` | Qdrant container tanımı | Volume/port davranışını etkiler |
| `web/demo/` | Tarayıcı istemcisi | Sunum katmanı |
| `docs/` | Mimari, yöntem ve final belgeleri | Okuma/raporlama |

## Config ve veri

| Dosya | İçerik |
|---|---|
| `configs/final_50k.json` | hedef 50K; API sorgu/yıl/sayfa; TR-MTEB; UMAP; Leaf sweep; Qdrant adları |
| `configs/retrieval_benchmark_queries.json` | Q01–Q12 ve metadata relevance grupları |
| `data/processed/final_articles_50000.jsonl` | her satır bir final makale; ID/yıl/title/abstract/keyword/database/subject |
| `data/processed/pilot_articles_1000.jsonl` | Day deneylerinin pilot seti |
| `data/processed/validation_articles_5000.jsonl` | Day29–31 bağımsız validation seti |
| `data/state/final_50k_checkpoint.json` | API toplama resume state |

## Ortak Python modülleri

| Dosya | Sorumluluk |
|---|---|
| `api_client.py` | HTTP istek, retry, TR Dizin sayfası |
| `dataset.py` | kayıt parse/normalize, geçerli abstract ve duplicate kontrolleri |
| `validation.py` | dataset kalite ölçüleri ve placeholder denetimi |
| `embeddings.py` | model yükleme, CUDA/CPU, normalize embedding |
| `clustering.py` | UMAP, HDBSCAN, centroid ve atama yardımcıları |
| `evaluation.py` | purity, overlap, silhouette gibi ölçüler |
| `qdrant_store.py` | REST collection/index/search işlemleri |
| `hybrid_search.py` | dense/sparse sonuçları ve RRF |
| `reporting.py` | final tablo/şekil/Markdown üretimi |
| `config.py`, `io_utils.py` | yol/ayar ve güvenli okuma-yazma yardımcıları |

## Final scriptler

| Script | Girdi → çıktı | Donanım/not |
|---|---|---|
| `01_collect_articles.py` | API → final JSONL + checkpoint | ağ/CPU; `--resume` destekli |
| `02_validate_dataset.py` | JSONL → kalite JSON/CSV/MD + iki PNG | CPU |
| `03_build_embeddings.py` | abstract_tr → 50K×768 `.npy` + metadata | varsayılan CUDA |
| `04_discover_topics.py` | embedding → UMAP/HDBSCAN/KMeans/atama/figür | CPU |
| `05_build_final_report.py` | final çıktılar → topic report | CPU |
| `06_index_qdrant.py` | abstract vectors+payload → ana collection | Qdrant yazımı; `--recreate` dikkat |
| `scripts/search/07_semantic_search.py` | query → dense Top-N | query embedding GPU |
| `08_retrieval_benchmark.py` | 12 sorgu → CSV/JSON/MD | Qdrant + model |
| `09_build_title_embeddings.py` | title_tr → title vectors | GPU |
| `10_index_title_qdrant.py` | title vectors → title collection | Qdrant yazımı |
| `11_hybrid_rrf_search.py` | abstract+title → iki-yol RRF | deneysel |
| `12_index_bm25_qdrant.py` | title+keyword → sparse collection | Qdrant yazımı |
| `13_three_way_hybrid_search.py` | abstract+title+BM25 → RRF | deneysel |
| `scripts/demo/14_demo_server.py` | HTTP API + static web | model bellekte; CUDA varsayılan |

## Final çıktılar

### `outputs/final_50k/embeddings/`

- `tr_mteb_50000.npy`: final abstract vektörleri, 50K×768.
- `tr_mteb_50000_metadata.json`: model, hash, GPU, shape, normalizasyon.
- `tr_mteb_titles_50000.npy` ve metadata: deneysel title dense.
- `aborted_before_dataset_repair_*`: eski/yarım koşu; final kanıt olarak kullanılmaz.

### `outputs/final_50k/clustering/`

- `hdbscan_parameter_sweep.csv`: altı 50K Leaf adayı.
- `final_topic_pipeline_summary.json`: 404/direct/noise ve seçim gerekçesi.
- `final_cluster_assignments.csv`: makale bazlı primary/secondary/margin/provenance.
- `noise_fallback_assignments.csv`: 23.347 fallback ayrıntısı.
- `final_cluster_dictionary.csv`: metadata-derived topic adları ve temsilciler.
- `cluster_centroids.npy`: fallbackte kullanılan merkezler.
- `kmeans_baseline_assignments.csv`: final karşılaştırma baseline'ı.

### `outputs/final_50k/reports/` ve `figures/`

- `dataset_quality_summary.json`, `dataset_quality_by_year.csv`, `dataset_quality_report.md`.
- `FINAL_50000_TOPIC_DISCOVERY_REPORT.md`, `FINAL_SEARCH_AND_QDRANT_REPORT.md`.
- `abstract_length_distribution.png`, `token_length_distribution.png`.
- `cluster_size_distribution.png`, `parameter_comparison.png`.
- `umap_2d_clusters.png`, `umap_2d_direct_fallback.png`.

### `outputs/final_50k/search/`

- `qdrant_index_manifest.json`: `trdizin_articles_50000`, abstract 768D cosine.
- `qdrant_title_index_manifest.json`: title 768D cosine.
- `qdrant_bm25_manifest.json`: title+keyword sparse BM25.
- `retrieval_benchmark_results.csv`: 120 sıralı sonuç.
- `retrieval_benchmark_summary.csv/json`: sorgu ve aggregate metrikler.
- `RETRIEVAL_BENCHMARK_REPORT.md`: okunur rapor.

## Altyapı ve web

`infra/qdrant/compose.yaml`, Qdrant 1.19.0'ı localhost REST 6335, gRPC 6336 ve named volume `trdizin_qdrant_storage` ile tanımlar. `web/demo/index.html`, `scripts/demo/14_demo_server.py` backendinden health/search JSON'u alır; Semantic ve Hybrid Experimental modları, filtreler, result card ve UMAP bölümlerini sunar.

## Güvenli okuma sırası

Önce config → kalite özeti → embedding metadata → clustering summary → manifestler → retrieval summary okuyun. `.npy` satırlarının JSONL sırasıyla hizalı olduğunu ve tüm final artefactlarda dataset hash'in aynı olduğunu doğrulamadan yeniden indeksleme yapmayın. `--recreate`, collectionı yeniden oluşturabileceği için yalnız bilinçli bakımda kullanılmalıdır.
