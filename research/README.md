# Araştırma ve Deney Geçmişi

Bu klasör Day01–Day31 boyunca yapılan deneyleri, benchmarkları, model ve parametre
seçimlerini ve karar geçmişini korur. Buradaki scriptler final uygulama entry point'i
değildir; final çalışan sistem `src/trdizin_topic_pipeline/` ve `scripts/` altındadır.

- `experiments/`: Altı araştırma fazına ayrılmış, kronolojik Day01–Day31 kaynakları.
- `outputs/`: Bu deneylerin ürettiği tarihsel artefactlar.

Scriptlerdeki çalıştırılabilir yollar yeni `research/outputs/` konumuna güncellenmiştir.
Önceden üretilmiş JSON dosyalarının içindeki mutlak `embedding_path` gibi provenance
alanları tarihsel sonucu değiştirmemek için yeniden yazılmamıştır; eski çalışma
anındaki konumu göstermeleri beklenir.

`outputs/final_50k/` ve `outputs/smoke_test/` final sisteme aittir ve bilinçli olarak
bu klasörün dışında tutulur. 50K artefactları yeniden üretilmemeli veya silinmemelidir.
