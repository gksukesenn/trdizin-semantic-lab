# Proje Yapısı

Bu ayrım araştırma geçmişini, yeniden kullanılabilir final kodunu ve kullanıcıya
açılan adaptörleri birbirine karıştırmadan sunmak için vardır.

- `research/`: Day01–Day31 deneyleri ve değişmeden korunan deney çıktıları.
- `src/`: Kurulabilir `trdizin_topic_pipeline` paketi; algoritmalar ve use-case'ler.
- `scripts/`: Numarası final pipeline sırasını gösteren ince CLI entry point'leri.
- `web/`: Python paketinden bağımsız statik demo arayüzü ve assetleri.
- `tests/`: Servissiz unit testleri, Qdrant integration testleri ve smoke kontrolleri.
- `data/`: Pipeline girdileri, ham API cevapları, işlenmiş veri ve checkpoint state'i.
- `outputs/`: Yalnız final ve smoke çalışma artefactları; araştırma çıktıları burada değildir.

## Final pipeline sırası

`01 collect -> 02 validate -> 03 embeddings -> 04 topics -> 05 report -> 06–13 search/index`

Qdrant 06/10/12 indeksleme adımlarında devreye girer. Semantic, BM25 ve RRF/hybrid
sorguları `search/` çekirdeği üzerinden Qdrant collection'larına erişir.

## Mimari ve debug akışı

```text
Browser
   |
   v
web/routes.py
   |
   v
services/search_service.py
   |
   +-------------------+
   |                   |
   v                   v
Semantic          BM25 / RRF Hybrid
   |                   |
   +---------+---------+
             |
             v
search/qdrant_store.py ---> Qdrant
```

Demo isteğinde breakpoint sırası `DemoHandler.do_POST`, `SearchService.search`,
`QueryEncoder.encode`, `build_filter`, `multi_source_rrf`, `QdrantRestStore.query_*`
ve `format_*_results` şeklindedir. `web/app.py` yalnız
composition root ve sunucu yaşam döngüsünü; `schemas.py` HTTP payload sözleşmesini taşır.

Qdrant indeksleme command'ları ortak `indexing/helpers.py`, `payloads.py` ve
`validation.py` çekirdeğini kullanır; abstract, title ve BM25'e özgü batch/index
akışları ayrı indexer modüllerinde kalır. Retrieval benchmark metrikleri
`evaluation/retrieval.py`, rapor yazımı ise `reporting/retrieval_report.py` içindedir.
