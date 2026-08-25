# Pilot Yöntem Seçimi: Day01–Day31

TF-IDF, sözcük örtüşmesine dayalı açıklanabilir bir baseline olarak denendi; nihai semantik temsil olarak seçilmedi. Çok dilli ve Türkçe embedding adayları komşuluk kalitesi, token kesilmesi ve metadata tutarlılığıyla karşılaştırıldı. Abstract uzunluk analizi `max_seq_length=512` kararını destekledi. `trmteb/turkish-embedding-model-fine-tuned`, Türkçe semantik yapı için seçildi; embeddingler 768 boyutlu ve normalize edildi.

KMeans k=30 tam kapsama baseline'ı olarak tutuldu. UMAP 10D ardından HDBSCAN deneylerinde EOM bazı seedlerde az sayıda büyük kümeye çöktü. Leaf, daha kararlı cluster çözünürlüğü sağladı. Noise kayıtlarının atanmasında medoid ve normalize centroid karşılaştırıldı; centroid fallback daha uygun nihai yöntem olarak benimsendi. Clusterın gerçek temsilci makalesi için medoid ayrıca korunur.

Bağımsız 5.000 makale doğrulamasında beş seed özeti:

| Yöntem | Top-1 metadata tutarlılığı | Top-2 | ARI | NMI |
|---|---:|---:|---:|---:|
| KMeans k=30 | %51,83 ± %0,41 | %62,97 ± %1,10 | 0,542 ± 0,024 | 0,751 ± 0,012 |
| UMAP10 + HDBSCAN Leaf | %58,71 ± %0,84 | %70,45 ± %0,42 | 0,603 ± 0,023 | 0,845 ± 0,009 |

Buradaki Top-1/Top-2 değerleri accuracy değil, subject metadata tutarlılığıdır; subject alanı ground truth değildir. ARI/NMI koşular arası üyelik kararlılığını destekler. Day31 yorumlanabilirlik denetimi temsilci başlıkları, direct/fallback oranlarını ve düşük tutarlılıklı clusterları insan incelemesine açtı.

Nihai karar: ana keşif yöntemi `abstract_tr → TR-MTEB → UMAP 10D → HDBSCAN Leaf`; noise çözümü normalize centroid; KMeans k=30 tam kapsama baseline'ıdır. 50.000 ölçekte `min_cluster_size=10` doğrudan kopyalanmaz; 25/50/100 ve `min_samples=5/10` kontrollü karşılaştırılır.
