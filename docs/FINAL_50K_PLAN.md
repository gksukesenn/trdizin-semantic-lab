# Nihai 50.000 Makale Pipeline Çalıştırma Planı

Day01–Day31 araştırma günlüğü korunur. Nihai uygulama `src/trdizin_topic_pipeline` paketinde, kullanıcı girişleri `scripts` altında ve bütün ayarlar `configs/final_50k.json` içinde tutulur.

## VS Code terminal komutları

```bash
cd ~/trdizin-semantic-lab
source .venv/bin/activate

python scripts/pipeline/01_collect_articles.py --config configs/final_50k.json --resume
python scripts/pipeline/02_validate_dataset.py --config configs/final_50k.json
python scripts/pipeline/03_build_embeddings.py --config configs/final_50k.json
python scripts/pipeline/04_discover_topics.py --config configs/final_50k.json
python scripts/pipeline/05_build_final_report.py --config configs/final_50k.json
```

CUDA yoksa embedding sessizce CPU'ya düşmez. Bilinçli CPU çalıştırması için `--allow-cpu` gerekir. UMAP, klasik HDBSCAN ve scikit-learn KMeans CPU tabanlıdır.

## Güvenli smoke test

```bash
python -m compileall src/trdizin_topic_pipeline scripts
python scripts/pipeline/01_collect_articles.py --config configs/final_50k.json --dry-run --target 100
python scripts/pipeline/02_validate_dataset.py --config configs/final_50k.json --smoke-test
python scripts/pipeline/03_build_embeddings.py --config configs/final_50k.json --smoke-test
python scripts/pipeline/04_discover_topics.py --config configs/final_50k.json --smoke-test
```

Smoke çıktıları `outputs/smoke_test` altındadır; final dosyaları değiştirilmez. Tam toplama ve embedding uzun işlemlerdir ve otomatik başlatılmaz.
