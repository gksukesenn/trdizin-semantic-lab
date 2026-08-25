# TR Dizin 50K — Hocanın Sorabileceği 55 Soru

Her maddede **Kısa cevap** 10–20 saniyelik; **Detaylı cevap** devam sorusu içindir.

## Veri ve amaç

### 1. Bu proje neden yapıldı?
**Kısa cevap:** 50.000 Türkçe akademik yayında konu yapısını keşfetmek ve doğal dille arama yapmak için.  
**Detaylı cevap:** Hazır embedding modeliyle clustering, overlap inceleme, 2B görselleştirme ve Qdrant semantic search aynı tekrarlanabilir hattın iki ayrı kolu olarak kuruldu.

### 2. TR Dizin nedir?
**Kısa cevap:** Türkiye'deki akademik dergi ve makaleleri indeksleyen ulusal bibliyografik platformdur.  
**Detaylı cevap:** Bu projede platformun yayın arama API'si, Türkçe özet ve metadata kaynağı olarak kullanıldı; kalite veya doğruluk otoritesi olarak değil.

### 3. API nedir?
**Kısa cevap:** Yazılımların yapılandırılmış istek ve yanıtlarla konuşma arayüzüdür.  
**Detaylı cevap:** Endpoint'e sorgu/yıl/sayfa parametreleri gönderildi, JSON yanıtından ID, yıl, başlık, özet ve metadata çıkarıldı.

### 4. Neden PDF indirmediniz?
**Kısa cevap:** İhtiyaç duyulan Türkçe abstract API'de vardı; PDF erişim ve ayrıştırma riski gereksizdi.  
**Detaylı cevap:** PDF tam metin zenginliği sunardı ama telif/erişim, taranmış sayfa, layout ve OCR sorunları getirirdi. Projenin abstract tabanlı amacı için API daha denetlenebilirdi.

### 5. Örneklem random mı?
**Kısa cevap:** Hayır.  
**Detaylı cevap:** 12 geniş sorgu, 2008–2025 yıl ve sayfa sırasıyla API tarandı. Dolayısıyla sonuç TR Dizin evreninin olasılıklı rastgele örneği değildir.

### 6. Neden tam 50.000?
**Kısa cevap:** Proje gereksinimi en az 50.000 yayındı ve pipeline hedefte durdu.  
**Detaylı cevap:** Final kalite raporu 50.000 satır, ID ve özet benzersizliğini doğrular.

### 7. Metadata nedir?
**Kısa cevap:** Makaleyi tanımlayan yıl, başlık, subject gibi yardımcı alanlardır.  
**Detaylı cevap:** Metnin kendisi değildir; filtre, raporlama, topic adı ve keşifsel değerlendirmede kullanıldı.

### 8. Neden yalnız Türkçe abstract?
**Kısa cevap:** Tutarlı dilde, yeterli semantik içerikli ve API'de doğrudan bulunan ortak alan olduğu için.  
**Detaylı cevap:** Başlık kısa, keyword coverage %37,156; subject ise etikettir ve girdiye konursa keşfi yapay biçimde yönlendirir.

### 9. Subject embeddinge girdi mi?
**Kısa cevap:** Hayır.  
**Detaylı cevap:** `abstract_tr` tek embedding girdisidir. Subject sonradan isimlendirme ve metadata tutarlılığı kontrolünde kullanıldı.

### 10. Duplicate nasıl kontrol edildi?
**Kısa cevap:** Article ID ve normalize özet hash'iyle.  
**Detaylı cevap:** Final kalite çıktısı 50.000 benzersiz ID ve 50.000 benzersiz abstract SHA-256 bildirir.

### 11. Placeholder abstract neden problem?
**Kısa cevap:** `-`, `--`, boş veya “T.Öz Yok” semantik içerik taşımaz.  
**Detaylı cevap:** Böyle bir kayıt vektör üretse bile bilimsel içeriği temsil etmez ve yakınlıkları kirletir; validation bunları geçersiz sayar.

### 12. Dataset hash neden var?
**Kısa cevap:** Çıktıların aynı veri sürümüne ait olduğunu kanıtlamak için.  
**Detaylı cevap:** Final dataset, abstract/title embedding ve Qdrant manifestlerinde aynı SHA-256 `41cf…7257` bulunur.

## Embedding ve benzerlik

### 13. Embedding nedir?
**Kısa cevap:** Metni semantik yakınlığı yansıtmaya çalışan sayı vektörüne çevirir.  
**Detaylı cevap:** Aynı modelle üretilen yakın anlamlı metinlerin vektörleri cosine açısından yakın olma eğilimindedir; bu “model insan gibi anlıyor” iddiası değildir.

### 14. Vektör nedir?
**Kısa cevap:** Sıralı bir sayı listesidir.  
**Detaylı cevap:** Her makale 768 float32 sayı; 50K veri `(50000,768)` matrisidir.

### 15. Neden 768 boyut?
**Kısa cevap:** Seçilen TR-MTEB modelinin çıktı boyutudur.  
**Detaylı cevap:** 768 tek tek adlandırılmış topic veya elle seçilen özellik değildir.

### 16. Cosine similarity nedir?
**Kısa cevap:** İki vektörün yönlerinin ne kadar benzer olduğunu ölçer.  
**Detaylı cevap:** Normalize vektörlerde nokta çarpımına eşittir. Skor kalibre edilmiş doğruluk veya probability değildir.

### 17. Neden normalize ettiniz?
**Kısa cevap:** Vektör büyüklüğü yerine yön/anlam yakınlığını karşılaştırmak için.  
**Detaylı cevap:** Final metadata her vektörün normalize üretildiğini doğrular; centroidler de karşılaştırmadan önce normalize edilir.

### 18. Modeli siz mi eğittiniz?
**Kısa cevap:** Hayır; hazır modelle inference yapıldı.  
**Detaylı cevap:** Proje yeni algoritma/model geliştirme değil; uygun hazır model seçme ve sistem kurma çalışmasıdır.

### 19. Neden dört model denendi?
**Kısa cevap:** Kalite, Türkçe uygunluk, hız, boyut ve bellek trade-off'unu görmek için.  
**Detaylı cevap:** MiniLM hızlı, E5 Top-1'de az üstün, GTE uzun contextli; TR-MTEB dengeli seçimdi.

### 20. TR-MTEB her metrikte en iyi miydi?
**Kısa cevap:** Hayır.  
**Detaylı cevap:** E5 pilot Top-1 exact'te %59,89 ile TR-MTEB'in %58,82'sini geçti; GTE Top-5 any exact'te az üstündü. TR-MTEB trade-off ile seçildi.

### 21. Neden E5-large değil?
**Kısa cevap:** Pilot kalite farkına karşı çok daha yavaş ve bellek ağırdı.  
**Detaylı cevap:** 23,74 doc/s ve 2.244 MB peak; TR-MTEB 75,61 doc/s ve 506 MB peak verdi.

### 22. 512 token ne etkiliyor?
**Kısa cevap:** Daha uzun özetlerin son kısmı kesilebilir.  
**Detaylı cevap:** Finalde 3.839 kayıt (%7,678) 512 token üstündedir; kayıt silinmez ama son bilgi embeddinge girmeyebilir.

## Clustering

### 23. Classification ve clustering farkı?
**Kısa cevap:** Classification hazır etiketi tahmin eder; clustering etiketsiz grup keşfeder.  
**Detaylı cevap:** Bu projede subject hedef değişken değil; clusterlar abstract uzayından oluşur.

### 24. KMeans nasıl çalışır?
**Kısa cevap:** k merkez başlatır, noktaları en yakına atar, merkezleri günceller.  
**Detaylı cevap:** Yakınsama olana kadar tekrarlar ve herkesi mutlaka bir kümeye atar.

### 25. k=30 nereden geldi?
**Kısa cevap:** 1K pilot k sweep'te karşılaştırılan adaylar içinde dengeli baseline olarak.  
**Detaylı cevap:** k=30 cosine silhouette `0,09043` idi; “30 gerçek konu” anlamına gelmez.

### 26. Silhouette nedir?
**Kısa cevap:** Noktanın kendi kümesine yakınlığını en yakın diğer kümeye göre kıyaslar.  
**Detaylı cevap:** +1 iyi ayrım, 0 sınır, negatif alternatif kümeye ortalamada daha yakınlık sinyalidir.

### 27. Negatif silhouette yanlış makale mi?
**Kısa cevap:** Hayır.  
**Detaylı cevap:** Disiplinler arası çalışma, örtüşme veya KMeans'in zorunlu ataması olabilir. Pilotta 204/1.000 negatiftir.

### 28. Neden KMeans yetmedi?
**Kısa cevap:** Herkesi atar, noise/geçiş alanını doğal biçimde göstermez.  
**Detaylı cevap:** Baseline olarak değerlidir; fakat yoğunluk tabanlı keşif gri alan analizi için daha uygundur.

### 29. HDBSCAN nedir?
**Kısa cevap:** Yoğun ve kalıcı bölgeleri bulan, bazı noktaları noise bırakabilen clustering yöntemidir.  
**Detaylı cevap:** Küme sayısını önceden istemez; yoğunluk hiyerarşisinden EOM veya Leaf seçim yapar.

### 30. Noise kötü veri mi?
**Kısa cevap:** Hayır.  
**Detaylı cevap:** Yalnız bu parametrelerle yeterince yoğun çekirdeğe direct bağlanmamıştır; geçiş alanı veya seyrek bir konu olabilir.

### 31. min_cluster_size nedir?
**Kısa cevap:** Küme sayılacak grubun asgari büyüklüğüdür.  
**Detaylı cevap:** Final sweep 25, 50, 100 değerlerini karşılaştırdı; 25 seçildi.

### 32. min_samples nedir?
**Kısa cevap:** Yoğunluk ölçütünün katılığını etkiler.  
**Detaylı cevap:** Genelde yükseldikçe daha temkinli direct atama ve daha fazla noise görülür; finalde 5 seçildi.

### 33. EOM ve Leaf farkı?
**Kısa cevap:** EOM daha kalıcı/geniş, Leaf daha ince uç kümeleri seçer.  
**Detaylı cevap:** EOM seedlerde collapse gösterdi; Leaf daha kararlı konu çözünürlüğü verdi.

### 34. H01/H16/H18 neydi?
**Kısa cevap:** 1K pilot sweep'teki üç HDBSCAN adayıydı.  
**Detaylı cevap:** H01 EOM 10/5, H16 Leaf 10/10, H18 Leaf 15/10. Final 50K parametreleri değillerdir.

### 35. H01 neden önce güçlüydü sonra elendi?
**Kısa cevap:** Pilot coverage/metadata iyi, holdout stability kötüydü.  
**Detaylı cevap:** Day27'de 800 train iki kümeye collapse oldu; Day28 EOM 5 seedin 4'ünde collapse etti.

### 36. Holdout nedir?
**Kısa cevap:** Karar kurulurken görülmeyip sonra test edilen ayrılmış veri.  
**Detaylı cevap:** 800 train cluster/ad, 200 test değerlendirme; test subjectleri sona kadar kullanılmadı.

### 37. Seed nedir?
**Kısa cevap:** Rastgele sürecin tekrarlanabilir başlangıç sayısıdır.  
**Detaylı cevap:** Seed değişimi UMAP ve başlangıçları etkileyebilir; bu yüzden 11,22,33,42,55 karşılaştırıldı.

### 38. Cluster collapse nedir?
**Kısa cevap:** Yöntemin ayrıntılı yapı yerine birkaç dev kümeye düşmesidir.  
**Detaylı cevap:** Yüksek direct coverage böyle durumda yanıltıcı olabilir; Day27'nin iki kümesi kritik uyarıdır.

### 39. ARI ve NMI ne ölçer?
**Kısa cevap:** Farklı koşulardaki üyelik yapısının benzerliğini, cluster numarasından bağımsız ölçer.  
**Detaylı cevap:** Leaf 5K'da ARI `0,603`, NMI `0,845`; KMeans `0,542`, `0,751` verdi. Bunlar topic doğruluğu değildir.

### 40. Neden cluster 3'ü cluster 3'le kıyaslamadınız?
**Kısa cevap:** Cluster numaraları koşular arasında keyfidir.  
**Detaylı cevap:** ARI/NMI üyelik ilişkilerini label permutationdan bağımsız karşılaştırır.

### 41. Neden PCA değil UMAP?
**Kısa cevap:** Bu veride PCA-HDBSCAN çok az kümeye ve yüksek noise'a çöktü.  
**Detaylı cevap:** PCA doğrusal varyansı, UMAP yerel komşulukları önceler; seçim teori kadar çok-seed çıktısına dayanır.

### 42. Neden clustering 10D?
**Kısa cevap:** Yerel yapıyı koruyup density clusteringi kararlı/uygulanabilir kılan doğrulanmış ara boyut.  
**Detaylı cevap:** 2D'den daha fazla bilgi taşır; 768D doğrudan HDBSCAN yerine pilot/stability mimarisi kullanıldı.

### 43. Neden görsel ayrı 2D?
**Kısa cevap:** İnsan ekranında görebilsin diye.  
**Detaylı cevap:** Ayrı fit edilir ve clustering girdisi değildir; eksenleri doğrudan semantik anlam taşımaz.

### 44. Neden aramada UMAP yok?
**Kısa cevap:** Semantic search en zengin orijinal 768D temsili kullanır.  
**Detaylı cevap:** UMAP indirgeme kaybı ve eğitim-bağımlı dönüşüm getirir; Qdrant normalize 768D cosine arar.

### 45. 404 gerçek konu mu?
**Kısa cevap:** Hayır.  
**Detaylı cevap:** 50K veri + TR-MTEB + UMAP ayarları + Leaf 25/5 altında bulunan density kümeleridir; parametre değişince sayı 109–404 aralığında değişti.

### 46. Centroid nedir?
**Kısa cevap:** Küme vektörlerinin normalize ortalama yönüdür.  
**Detaylı cevap:** Gerçek makale olmak zorunda değildir; fallback için her noise kaydı en yakın centroidle eşleştirildi.

### 47. Centroid fallback hile değil mi?
**Kısa cevap:** Hayır; açıkça işaretlenmiş ayrı bir tam-kapsama katmanıdır.  
**Detaylı cevap:** Direct üyelik diye sunulmaz. Day24'te alternatif temsil yöntemlerine göre metadata Top-1 ve H01 recovery daha yüksekti.

### 48. Primary/secondary nasıl üretildi?
**Kısa cevap:** Birinci ve ikinci konu/centroid yakınlığıdır.  
**Detaylı cevap:** Direct kaydın primary'si HDBSCAN clusterı; fallback'in primary'si en yakın centroidtir. Secondary en yakın alternatif clusterdır.

### 49. Margin confidence mı?
**Kısa cevap:** Hayır.  
**Detaylı cevap:** `primary_similarity-secondary_similarity`; kalibre edilmiş olasılık değildir ve 0,03 “%3 emin” demek değildir.

### 50. Subject ground truth değilse neden kullandınız?
**Kısa cevap:** Bağımsız olmayan ama yararlı bir keşifsel tutarlılık sinyali olduğu için.  
**Detaylı cevap:** Çok etiketli ve indeksleme amaçlıdır; bu yüzden insan denetimi, ARI/NMI ve stability ile birlikte okunur.

## Arama ve demo

### 51. Qdrant neden gerekli?
**Kısa cevap:** 50K 768D vektörü hızlı aramak ve metadata filtresi uygulamak için.  
**Detaylı cevap:** `.npy` brute force mümkün olsa da servis, payload indexi, kalıcı collection ve API sağlamaz.

### 52. Neden normal SQL değil?
**Kısa cevap:** SQL kesin alan eşleşmesinde, Qdrant vektör yakınlığında uzmanlaşır.  
**Detaylı cevap:** Yıl/database filtreleri payload, anlam yakınlığı cosine search ile aynı istekte birleşir.

### 53. Benchmark sonuçlarına neden accuracy demediniz?
**Kısa cevap:** 12 sorgu ve subject-string relevance gerçek insan ground truth'u değildir.  
**Detaylı cevap:** P@5/P@10, MRR ve nDCG sıralama ölçüleridir; ortalamalar keşifsel metadata tutarlılığıdır.

### 54. BM25 ve RRF neden eklendi?
**Kısa cevap:** Dense aramanın kaçırdığı teknik kelime niyetini title/keyword ile desteklemek için.  
**Detaylı cevap:** RRF farklı skor ölçeklerini ham ağırlıklarla karıştırmadan abstract dense, title dense ve BM25 sıralarını birleştirir.

### 55. Hybrid neden final olmadı; demo ne gösteriyor?
**Kısa cevap:** Q03'e yardım etti ama Q10'u kesin çözmedi; ortak insan etiketli üstünlük kanıtı yok.  
**Detaylı cevap:** Ana mod abstract dense'dir; “Hybrid Experimental” açıkça deneysel kalır. Karttaki score olasılık değil; Direct/Fallback provenance, cluster, primary/secondary ve margin ayrı açıklanır.

## Sunumda unutulmaması gereken tek cümle

“Bu sistem hazır TR-MTEB ile abstractların semantik uzayını kurar; clustering keşif, Qdrant arama görevidir; 404 cluster ontoloji, similarity olasılık ve subject ground truth değildir.”
