# TR Dizin Semantic Lab

TR Dizin'deki 50.000 Türkçe makale için konu keşfi ve hybrid retrieval sistemi.
Repository araştırma karar geçmişi ile final çalışan sistemi kesin sınırlarla ayırır:

- `research/experiments/`: Day01–Day31 araştırma scriptleri.
- `research/outputs/`: Küçük, paylaşılabilir deney özetleri ve görselleri.
- `src/trdizin_topic_pipeline/`: Kurulabilir, sorumluluk bazlı final Python paketi.
- `scripts/`: Sıralamayı gösteren ince CLI entry point'leri.
- `web/demo/`: Statik demo frontend'i (`assets/css`, `assets/js`).
- `outputs/final_50k/`: Yerelde tutulan, Git yedeğine alınmayan final artefactlar.
- `outputs/smoke_test/`: Yerelde yeniden üretilebilen doğrulama artefactları.

Ayrıntılı mimari ve mentor sunumu rehberi: [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).
Bilimsel karar geçmişi: [docs/TRDIZIN_50000_DENEY_VE_KARAR_RAPORU.md](docs/TRDIZIN_50000_DENEY_VE_KARAR_RAPORU.md).

## GitHub'dan temiz kurulum

Bu depo, 50.000 TR Dizin makalesi üzerinde embedding, konu keşfi ve
semantic/hybrid search deneyleri geliştirmek için kullanılan kodu ve yeniden
üretilebilir yapılandırmaları içerir.

```bash
git clone https://github.com/<kullanici>/trdizin-semantic-lab.git
cd trdizin-semantic-lab
cp .env.example .env

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

cd infra/qdrant
docker compose up -d --build
cd ../..
```

Docker Compose bu projede Qdrant servisini başlatır. Varsayılan REST adresi
[http://127.0.0.1:6335](http://127.0.0.1:6335)'tir. Alternatif olarak bağımlılıklar
`python -m pip install -r requirements.txt` ile kurulabilir.

Büyük ham/işlenmiş makale verileri, üretilmiş embedding dosyaları ve yerel Qdrant
verisi güvenlik ve boyut nedenleriyle depoya dahil edilmez. Bu nedenle temiz bir
klonda 50K koleksiyonlarını kullanmadan önce veriyi yetkili kaynaktan ayrıca
sağlamak, pipeline ile embedding'leri üretmek ve Qdrant indeksleme scriptlerini
çalıştırmak gerekir.

## Yerel kurulum

```bash
python -m pip install -e .
```

`requirements.txt` mevcut kurulum akışları için korunmuştur. Editable kurulumdan sonra
scriptler çalışma dizinine özel `sys.path` müdahalesi olmadan paketi import eder.

## Final pipeline

```bash
python scripts/pipeline/01_collect_articles.py --help
python scripts/pipeline/02_validate_dataset.py --help
python scripts/pipeline/03_build_embeddings.py --help
python scripts/pipeline/04_discover_topics.py --help
python scripts/pipeline/05_build_final_report.py --help
python scripts/search/07_semantic_search.py --help
python scripts/demo/14_demo_server.py --help
```

Numaralandırma çalıştırma ve sunum sırasıdır; algoritma kodu scriptlerde değil pakettedir.
Demo için Qdrant collection'ları gerekir; CPU ile başlatmak için `--allow-cpu` verilebilir.
Uzun embedding/clustering adımları final artefactları doğrulamak amacıyla yeniden çalıştırılmamalıdır.

## Doğrulama

```bash
python -m compileall -q src scripts research/experiments tests
pytest -q
```

Gerçek Qdrant integration testi varsayılan olarak skip edilir; açıkça çalıştırmak için
`TRDIZIN_QDRANT_INTEGRATION=1` ayarlanır. VS Code launch profilleri `.vscode/launch.json`
altındadır.

## Backup / reproducibility notes

Bu private yedek kaynak kodunu, deney scriptlerini, testleri, yapılandırmaları,
dokümantasyonu ve küçük araştırma özetlerini korur. `.env`, `data/` altındaki ham
ve işlenmiş içerikler, `outputs/` runtime artefactları, embedding/model ikilileri,
cache/log dosyaları ve Docker volume'ları özellikle yedeğin dışındadır.

Temiz klonda Python ortamını kurduktan sonra `infra/qdrant` içinden
`docker compose up -d --build` ile Qdrant'ı başlatın. Ardından gerekli veri ve
embedding'leri yeniden üretip `scripts/search/06_index_qdrant.py` ve ihtiyaç
duyulan diğer indeksleme scriptleriyle yerel Qdrant koleksiyonlarını oluşturun.
