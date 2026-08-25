"""Nihai Türkçe Markdown rapor üretimi."""
import json
from pathlib import Path
from typing import Any, Dict
from ..utils.io import atomic_text

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists(): raise FileNotFoundError("Rapor girdisi bulunamadı: %s" % path)
    with path.open("r", encoding="utf-8") as handle: value = json.load(handle)
    if not isinstance(value, dict): raise ValueError("Rapor girdisi nesne değil: %s" % path)
    return value

def build_report(config: Dict[str, Any], quality: Dict[str, Any], topics: Dict[str, Any], target: Path) -> None:
    selected = topics.get("selected_parameters", {}); count = int(topics.get("dataset_row_count", quality.get("row_count", 0)))
    text = """# 50.000 Makalede Türkçe Konu Keşfi Nihai Raporu

## 1. Araştırma sorusu

TR Dizin'deki Türkçe bilimsel makale abstractları, önceden etiket zorunluluğu olmadan hangi semantik konu yapılarını gösterir?

## 2. Veri kaynağı ve kabul ölçütleri

Kaynak TR Dizin arama API'sidir. Yalnız PAPER, TUR, article_id ve Türkçe abstract içeren kayıtlar kabul edilmiştir.

## 3. Veri toplama ve deduplication

Yıl, sorgu ve sayfa eksenlerinde çeşitlendirilmiş istekler kullanılmış; pilot ve validation ID'leri dışlanmıştır. Article ID ve normalize abstract SHA-256 tekrarları engellenmiştir.

## 4. Veri seti kapsamı

- Makale: {count}
- Benzersiz ID: {ids}
- Benzersiz abstract: {abstracts}
- Boş başlık: {empty_titles}
- Subject bulunan oran: %{subject_rate:.2f}
- 512 token üzeri: {over512} (%{over512_rate:.2f})
- Dataset SHA-256: `{dataset_hash}`

## 5. Pilot yöntem seçimi

Day01–Day31 deney günlüğü TF-IDF baseline'ından embedding karşılaştırmasına, kararlılık ve bağımsız 5.000 makale doğrulamasına uzanır. Ayrıntı `docs/PILOT_METHOD_SELECTION.md` dosyasındadır.

## 6. Neden TR-MTEB?

Türkçe semantik komşuluk ve metadata tutarlılığı deneyleri sonunda `trmteb/turkish-embedding-model-fine-tuned` seçilmiştir. Girdi yalnız `abstract_tr`, boyut 768 ve normalizasyon açıktır.

## 7. Neden HDBSCAN Leaf?

EOM pilot koşularında çözünürlük çökmesi görülürken Leaf daha kararlı ve yorumlanabilir konu ayrımı sağlamıştır.

## 8. KMeans baseline

KMeans k={kmeans_k} tam kapsama baseline'ıdır. Weighted subject metadata tutarlılığı {kmeans_purity:.4f} olarak ölçülmüştür; bu değer accuracy değildir.

## 9. 50.000 ölçeğinde parametre kontrolü

Seçim durumu: **{selection_status}**. Seçilen `min_cluster_size={mcs}`, `min_samples={ms}`, `cluster_selection_method=leaf`. Gerekçe: {selection_reason}

Tek bir composite score kullanılmamış; çökme, noise, çözünürlük, küçük/büyük cluster, metadata tutarlılığı ve silhouette ayrı ayrı incelenmiştir.

## 10. Nihai cluster sonuçları

- Cluster: {clusters}
- Direct atama: {direct}
- Noise/fallback: {noise} (%{noise_rate:.2f})

## 11. Noise ve centroid fallback

HDBSCAN noise makaleleri konusuz değildir. Normalize 768D centroidlere cosine benzerliğiyle birincil ve farklı ikincil konu atanmıştır.

## 12. Birincil/ikincil konu çıktısı

Her satır primary/secondary cluster, benzerlik ve margin içerir. Margin kalibre edilmiş olasılık veya güven skoru değildir.

## 13. Çok alanlı ve gri alan örnekleri

Düşük marginli kayıtlar semantik gri alan incelemesi için adaydır. Subject metadata ground truth değildir; yalnız keşifsel adlandırma ve tutarlılık kontrolünde kullanılmıştır.

## 14. UMAP görselleştirmeleri

- [UMAP 2D cluster görünümü](../figures/umap_2d_clusters.png)
- [UMAP 2D direct/fallback görünümü](../figures/umap_2d_direct_fallback.png)
- [Cluster boyutları](../figures/cluster_size_distribution.png)
- [Parametre karşılaştırması](../figures/parameter_comparison.png)

Clustering UMAP 10D üzerinde yapılmıştır. UMAP 2D yalnız görselleştirmedir ve boyut indirgeme nedeniyle bilgi kaybettirir.

## 15. Semantic search için hazır veri yapısı

JSONL sıra indeksi embedding sıra indeksiyle hizalıdır; normalize embedding, centroid ve assignment tabloları gelecekte arama indeksine aktarılabilir.

## 16. Sınırlamalar

API sorgu kapsamı tüm evrenin rastgele örneklemi değildir. Subject metadata eksik/hiyerarşik olabilir ve ground truth değildir. Topic adları geçicidir.

## 17. Tekrarlanabilirlik

Config: `{config_path}`; random seed: `{seed}`. Model `{model}`, max_seq_length={max_seq}, normalize_embeddings=true. UMAP n_neighbors={neighbors}, min_dist={min_dist}, metric={metric}.

## 18. Sonuç ve öneriler

UMAP 10D + HDBSCAN Leaf ana keşif yöntemi, KMeans ise tam kapsama baseline'ı olarak korunmuştur. İnsan denetimi özellikle düşük marginli ve metadata tutarlılığı düşük clusterlarda sürdürülmelidir.
""".format(count=count, ids=quality.get("unique_article_id_count", 0), abstracts=quality.get("unique_abstract_sha256_count", 0),
 empty_titles=quality.get("empty_title_count", 0), subject_rate=100*quality.get("subject_present_rate", 0), over512=quality.get("over_512_token_count", 0), over512_rate=100*quality.get("over_512_token_rate", 0), dataset_hash=quality.get("dataset_sha256", ""),
 kmeans_k=config["kmeans"]["n_clusters"], kmeans_purity=topics.get("kmeans", {}).get("weighted_subject_purity", 0), selection_status=topics.get("selection_status", "karar_verilemedi"), mcs=selected.get("min_cluster_size", "-"), ms=selected.get("min_samples", "-"), selection_reason=topics.get("selection_reason", ""), clusters=topics.get("cluster_count", 0), direct=topics.get("direct_count", 0), noise=topics.get("noise_count", 0), noise_rate=100*topics.get("noise_rate", 0), config_path=config.get("_config_path", ""), seed=config["random_seed"], model=config["embedding"]["model_id"], max_seq=config["embedding"]["max_seq_length"], neighbors=config["umap"]["n_neighbors"], min_dist=config["umap"]["min_dist"], metric=config["umap"]["metric"])
    atomic_text(target, text)
