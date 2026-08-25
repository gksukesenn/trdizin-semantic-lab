# TR Dizin 50K — Yeni Stajyer Cheat Sheet

## Bir cümlede proje

TR Dizin API'den toplanan 50.000 Türkçe makale özeti, hazır TR-MTEB modeliyle 768 boyutlu normalize vektörlere çevrildi; UMAP10+HDBSCAN Leaf ile konu kümeleri keşfedildi ve aynı orijinal vektörler Qdrant'ta semantik aramaya açıldı.

```text
50K makale → abstract_tr → TR-MTEB 768D → UMAP 10D
                                           → HDBSCAN Leaf
                                           → 404 cluster
                                           → direct / fallback

TR-MTEB 768D → Qdrant Cosine → semantic search
```

## Rakamlarla final

| Alan | Değer |
|---|---:|
| Makale / benzersiz ID / benzersiz özet | 50.000 / 50.000 / 50.000 |
| Dönem | 2008–2025 |
| Subject coverage | %99,83 |
| Keyword coverage | %37,156 |
| 512 token üstü | 3.839 (%7,678) |
| Embedding | TR-MTEB, normalize float32, `(50000,768)` |
| Clustering | ayrı UMAP 10D → HDBSCAN Leaf, mcs=25, ms=5 |
| Cluster | 404 |
| Direct / centroid fallback | 26.653 / 23.347 |
| Abstract Qdrant | 50.000 point, 768D, Cosine |
| Benchmark | P@5=0,85; P@10=0,85; MRR=0,854; nDCG=0,871 |

Kaynaklar: `outputs/final_50k/reports/dataset_quality_summary.json`, `outputs/final_50k/clustering/final_topic_pipeline_summary.json`, Qdrant manifestleri, retrieval benchmark özeti.

## Kavramlar

- **Embedding:** Metni, semantik karşılaştırmaya uygun sayı listesine dönüştüren hazır model çıktısı.
- **768D:** Makale başına 768 sayı. 768 konu değildir.
- **Cosine:** Vektör yönlerinin benzerliği. Probability değildir.
- **Clustering:** Hazır etiket olmadan grup keşfi.
- **KMeans:** k ister ve herkesi atar; projede k30 baseline.
- **UMAP:** Yerel komşulukları daha az boyutta yaklaşık korur.
- **HDBSCAN:** Yoğun bölgeleri bulur, bazı kayıtları noise bırakır.
- **Noise:** Direct yoğunluk kümesine girmeyen kayıt; kötü veri değildir.
- **Centroid:** Küme vektörlerinin ortalama yönü.
- **Fallback:** Noise kaydını en yakın centroidle sonradan tam kapsama atama.
- **Margin:** İlk ve ikinci centroid similarity farkı; confidence değildir.
- **Qdrant:** Vektör yakınlığı + payload filtresi yapan veritabanı.
- **BM25:** Exact/ayırt edici kelime sinyalli lexical arama.
- **RRF:** Farklı arama listelerini sıra üzerinden birleştirme.

## Neden bu seçimler?

| Seçim | Gerekçe |
|---|---|
| API, PDF değil | Türkçe özet yapılandırılmış; PDF ayrıştırma/erişim amaç dışı |
| Yalnız abstract | Zengin ortak metin; subject leakage yok |
| TR-MTEB | Türkçe kalite, hız, bellek ve 768D dengesi |
| Cosine + normalizasyon | Vektör yön/anlam yakınlığı |
| UMAP 10D | Leaf için çok-seed ve 5K validation desteği |
| Ayrı UMAP 2D | Sadece insan görselleştirmesi |
| Leaf, EOM değil | EOM seedlerde collapse; Leaf kararlı |
| Centroid fallback | Day24'te medoid/core average'dan daha iyi tam kapsama göstergesi |
| Aramada orijinal 768D | UMAP bilgi kaybını aramaya taşımamak |
| Qdrant | 50K vector search + yıl/database/topic filtreleri |

## Deneysel karar öyküsü

1. KMeans k30 pilotta ortalama silhouette `0,09043`, 204 negatif kayıt verdi.
2. HDBSCAN sweep'te H01/H16/H18 trade-off adayları oluştu.
3. H01 coverage/metadata ile bir ara güçlü göründü.
4. 800/200 holdout H01/EOM'u 2 kümeye collapse etti; Top-1 `0,0722`.
5. Beş seedte UMAP10 Leaf 30,0±1,67 küme ve sıfır collapse verdi.
6. Bağımsız 5K'da Leaf Top-1 `0,5871`, Top-2 `0,7045`, ARI `0,6033`, NMI `0,8449`; KMeans'ten yüksekti.
7. Final 50K sweep Leaf 25/5'i seçti: 404 direct cluster.
8. Dense retrieval Q03/Q10'da zorlandı; title dense+BM25+RRF denendi ama deneysel kaldı.

## Direct ve fallback'i okuma

`Direct HDBSCAN`: makale 10D yoğunluk yapısında clusterın doğal üyesi.  
`Centroid Fallback`: HDBSCAN noise; ürün tam kapsaması için 768D'de en yakın centroid atanmış.  
İkisi arayüzde özellikle ayrılır. Fallback kesin belirsiz, direct kesin doğru değildir.

## Clustering ve search aynı şey değil

| Clustering | Semantic search |
|---|---|
| Sorgusuz veri yapısı keşfi | Kullanıcı sorgusuna Top-N |
| UMAP 10D + HDBSCAN | orijinal normalize 768D + Qdrant |
| 404 yapılandırmaya bağlı grup | cosine sıralı makaleler |
| topic/overlap keşfi | erişim ve filtreleme |

## Arama modları

**Final:** query → TR-MTEB 768D → abstract Qdrant cosine.  
**Deneysel:** abstract dense + title dense + BM25 → RRF.

Q03 “yapay zekânın eğitimde kullanılması” birleşik niyeti dense sıralamada dağınık kaldı. Q10 “Türkçe doğal dil işleme ve metin sınıflandırma” genel Türk dili yayınlarına kaydı; teknik NLP yayınları corpus'ta bulunduğu halde ilk 10'a gelmedi. Bu nedenle ortalama benchmark tek başına yeterli değildir.

## Asla böyle söyleme

- “404 gerçek konu bulduk.” → “Bu yapılandırmada 404 density cluster oluştu.”
- “Model %85 doğru.” → “12 sorguda metadata tabanlı ortalama P@5 0,85.”
- “Noise kötü veridir.” → “HDBSCAN direct yoğunluk üyeliği vermedi.”
- “Margin confidence.” → “İlk iki cosine benzerliği farkı.”
- “Subject gerçek etiket.” → “Yardımcı, çok etiketli indeks metadata'sı.”
- “2D'de clusterladık.” → “10D'de clusterladık; ayrı 2D yalnız görsel.”

## Nereden başlamalı?

1. `configs/final_50k.json`
2. `outputs/final_50k/reports/dataset_quality_report.md`
3. `outputs/final_50k/reports/FINAL_50000_TOPIC_DISCOVERY_REPORT.md`
4. `outputs/final_50k/reports/FINAL_SEARCH_AND_QDRANT_REPORT.md`
5. `scripts/pipeline/04_discover_topics.py`, `scripts/search/07_semantic_search.py`, `scripts/demo/14_demo_server.py`
6. Ana rapor: [TRDIZIN_50000_KAPSAMLI_FINAL_RAPOR.md](TRDIZIN_50000_KAPSAMLI_FINAL_RAPOR.md)

