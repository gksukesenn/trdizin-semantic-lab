# TR Dizin 50K — Deney Kronolojisi

> Ölçek anahtarı: Day08–28 ağırlıkla 1K pilot; Day29–31 bağımsız 5K validation; `outputs/final_50k` 50K finaldir. H01/H16/H18 final 50K parametresi değildir.

| Gün/aşama | Soru veya hipotez | Yöntem/çıktı | Doğrulanmış sonuç | Sonraki karara etkisi |
|---|---|---|---|---|
| Day01 | Problem türleri nasıl ayrılır? | classification/clustering/retrieval çerçevesi | Kavramsal ayrım | İki ayrı final kol tasarlandı |
| Day02 | Embedding ne sağlar? | küçük örnek | semantik vektör mantığı | Dense temsil adayı |
| Day03 | TF-IDF ve embedding farkı? | küçük retrieval karşılaştırması | embedding eşanlam bağlamını daha iyi taşıyabilir | TF-IDF açıklanabilir baseline kaldı |
| Day04 | TR Dizin arama yanıtı nasıl? | API inceleme | publication endpoint/JSON yapısı | Toplayıcı tasarımı |
| Day05–06 | Makale ve detay alanları neler? | extract/detail inspection | Türkçe abstract ve metadata erişilebilir | PDF yerine API |
| Day07 | Subject coverage yeterli mi? | coverage çıktısı | değerlendirme için kullanılabilir | subject yalnız audit sinyali |
| Day08 | Pilot nasıl kurulur? | 1.000 makale | ayrı pilot ID seti | hızlı yöntem seçimi |
| Day09 | Token limit etkisi? | tokenizer analizi | TR-MTEB pilot truncation %9,1; MiniLM %96,8 | 512 tokenlı adaylar öne çıktı |
| Day10 | Model hız/bellek? | 200 doküman GPU benchmark | MiniLM 510,5; TR-MTEB 75,6; E5 23,7; GTE 52,5 doc/s | trade-off tablosu |
| Day11–12 | Komşular metadata ile tutarlı mı? | 187 anchor, Top-5 | TR-MTEB Top-1 exact %58,82; E5 %59,89 | TR-MTEB dengeli final model |
| Day13 | 1K embedding üretilebilir mi? | normalize 768D | sıra hizalı dosya | clustering girdisi |
| Day14 | KMeans k ne olsun? | k=5…50 sweep | TR-MTEB k30 silhouette 0,09043 | k30 baseline |
| Day15 | KMeans kümeleri okunabilir mi? | başlık/subject raporu | 204 negatif silhouette | overlap/noise yöntemi gerekli |
| Day16 | 2D görsel ne gösterir? | UMAP 2D | adalar ve örtüşme | görsel, clustering kanıtı değil |
| Day17 | Density cluster adayları? | 28 HDBSCAN config | H01/H16/H18 trade-off | üç aday shortlist |
| Day18 | Adaylar ayrıntılı nasıl? | aynı 1K karşılaştırması | H01 33/%23; H16 30/%33,4; H18 21/%39,8 | tek silhouette reddedildi |
| Day19 | H16 okunabilir mi? | cluster/noise raporu | yorumlanabilir çekirdekler | metadata audit ihtiyacı |
| Day20 | KMeans vs H16 subject? | aynı-altküme kontrolü | H16 purity 0,6568; aynı subset KMeans 0,6254 | HDBSCAN çekirdekleri desteklendi |
| Day21 | Direct+fallback pipeline? | hybrid topic atama prototipi | tam kapsama mümkün | provenance alanı gerekli |
| Day22 | H01/H16/H18 kalite? | common subset/coverage | H01 coverage ve gain güçlü; H18 zayıfladı | H01 ana aday oldu |
| Day23 | H01 pipeline? | EOM direct/noise | 770 direct/230 noise | fallback yöntemini seç |
| Day24 | Noise temsili? | medoid/centroid/core mean | centroid Top-1 0,3869, recovery 0,9636 | centroid fallback |
| Day25–26 | Pipeline ve yeni abstract? | H01 centroid/runtime | inference akışı | holdout gereksinimi |
| Day27 | Görülmemiş veride ne olur? | 800 train/200 test | train 2 cluster; Top-1 0,0722 | kritik collapse uyarısı |
| Day28 | Seed/method stability? | 5 seed, UMAP/PCA/EOM/Leaf/KMeans | UMAP Leaf sıfır collapse; EOM/PCA sorunlu | Leaf finalist |
| Day29 | Bağımsız ölçek? | 5.000 validation toplama/embedding | ayrı ID seti | daha güçlü seçim |
| Day30 | KMeans vs Leaf 5K? | 5 seed + ARI/NMI | Leaf Top-1 0,5871, ARI 0,6033; KMeans 0,5183/0,5416 | Leaf final yöntem |
| Day31 | İnsan açısından yorumlanabilir mi? | seed33 cluster audit | gerçek başlıklar, direct/fallback, warningler | insan denetimi ve sınırlamalar |
| Final veri | 50K kalite? | validate+SHA | 50K unique, subject %99,83 | final embeddinge izin |
| Final embedding | 50K temsil? | TR-MTEB CUDA | `(50000,768)`, normalize | iki kola ayrıldı |
| Final clustering | Parametre ölçeği? | Leaf mcs 25/50/100, ms 5/10 | 25/5: 404 cluster, 26.653 direct | final topic discovery |
| Qdrant | Online vector search? | 3 collection | her biri 50K point | semantic/hybrid servis |
| Retrieval | Dense nerede zorlanır? | 12 sorgu | ort. P@5 0,85; Q03 0,2; Q10 0 | failure analysis |
| Title/BM25/RRF | Teknik niyet güçlenir mi? | deneysel üç-yol fusion | Q03 kısmi iyileşme, Q10 tam çözülmedi | final dense; hybrid deneysel |
| Demo | Sonuç nasıl sunulur? | Python backend + HTML | provenance ve marj alanları | öğretici web arayüzü |

## Karar değişikliklerinin anlamı

H01'in bir ara seçilip sonra Leaf'e dönülmesi tutarsızlık değildir. Pilot tek veri ve seedde hipotez üretir; holdout genelleme riskini, çok-seed testi kararlılığı, 5K validation ölçek davranışını sınar. Her yeni aşama önceki kararın kapsamadığı bir riski ölçmüştür.

## Ana kaynaklar

`research/outputs/day09_*`, `day10_embedding_benchmark.csv`, `day12_subject_neighbor_summary.csv`, `day14_kmeans_sweep.csv`, `day17_hdbscan_sweep_summary.csv`, `day18_hdbscan_candidate_comparison.csv`, `day20_cluster_subject_quality_summary.csv`, `day22_hdbscan_config_decision.csv`, `day24_noise_assignment_method_summary.csv`, `day27_holdout_summary.json`, `day28_stability_method_summary.csv`, `day30_finalist_summary.csv`, `day30_pairwise_stability.csv`, `day31_cluster_interpretability_summary.csv`, `outputs/final_50k/**`.

