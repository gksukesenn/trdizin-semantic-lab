from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# 1. Küçük örnek veri setimiz
# ---------------------------------------------------------
#
# Gerçek projede burada TR Dizin'den gelen yaklaşık
# 50.000 Türkçe makale olacak.
#
# Bugün yalnızca kavramları öğrenmek için 12 örnek kullanıyoruz.
#

ARTICLES = [
    {
        "id": "A01",
        "text": (
            "Diferansiyel denklemlerin sayısal yöntemlerle çözümü "
            "ve çözüm hatalarının matematiksel analizi incelenmiştir."
        ),
        "labels": ["Matematik"],
    },
    {
        "id": "A02",
        "text": (
            "Lineer cebir, matris ayrışımları ve özdeğer hesaplama "
            "yöntemleri karşılaştırılmıştır."
        ),
        "labels": ["Matematik"],
    },
    {
        "id": "A03",
        "text": (
            "Olasılık dağılımları ve istatistiksel tahmin yöntemleri "
            "üzerine yeni bir matematiksel model önerilmiştir."
        ),
        "labels": ["Matematik"],
    },
    {
        "id": "A04",
        "text": (
            "Kuantum parçacıklarının davranışı dalga denklemleri "
            "ve matematiksel operatörler kullanılarak incelenmiştir."
        ),
        "labels": ["Fizik", "Matematik"],
    },
    {
        "id": "A05",
        "text": (
            "Genel görelilik teorisindeki uzay zaman geometrisi "
            "diferansiyel denklemler yardımıyla modellenmiştir."
        ),
        "labels": ["Fizik", "Matematik"],
    },
    {
        "id": "A06",
        "text": (
            "Yarı iletken malzemelerde elektron hareketi, enerji "
            "seviyeleri ve elektriksel iletkenlik araştırılmıştır."
        ),
        "labels": ["Fizik"],
    },
    {
        "id": "A07",
        "text": (
            "Derin öğrenme modeli kullanılarak görüntüler üzerinde "
            "nesne sınıflandırma sistemi geliştirilmiştir."
        ),
        "labels": ["Bilgisayar Bilimleri"],
    },
    {
        "id": "A08",
        "text": (
            "Türkçe metinler için doğal dil işleme ve metin "
            "sınıflandırma yöntemleri karşılaştırılmıştır."
        ),
        "labels": ["Bilgisayar Bilimleri"],
    },
    {
        "id": "A09",
        "text": (
            "Graf sinir ağları ve matematiksel çizge modelleri "
            "kullanılarak bağlantı tahmini yapılmıştır."
        ),
        "labels": ["Bilgisayar Bilimleri", "Matematik"],
    },
    {
        "id": "A10",
        "text": (
            "Depresyon hastalarında uygulanan psikolojik tedavi "
            "yöntemlerinin klinik sonuçları incelenmiştir."
        ),
        "labels": ["Tıp"],
    },
    {
        "id": "A11",
        "text": (
            "Kanser hücrelerinin erken tespiti için tıbbi görüntüleme "
            "yöntemleri değerlendirilmiştir."
        ),
        "labels": ["Tıp"],
    },
    {
        "id": "A12",
        "text": (
            "Kalp hastalıklarının tanısında kullanılan klinik "
            "belirteçler ve tedavi süreçleri analiz edilmiştir."
        ),
        "labels": ["Tıp"],
    },
]


def show_classification_examples() -> None:
    """Classification ile multi-label classification farkını gösterir."""

    print("\n" + "=" * 70)
    print("1. CLASSIFICATION VE MULTI-LABEL CLASSIFICATION")
    print("=" * 70)

    single_label_article = ARTICLES[0]
    multi_label_article = ARTICLES[3]

    print("\nClassification örneği:")
    print(f"Input : {single_label_article['text']}")
    print(f"Output: {single_label_article['labels'][0]}")

    print("\nMulti-label classification örneği:")
    print(f"Input : {multi_label_article['text']}")
    print(f"Output: {multi_label_article['labels']}")

    print(
        "\nNot: Bugün classification modeli eğitmiyoruz. "
        "Yalnızca bu problemin beklediği output biçimini görüyoruz."
    )


def create_vectors():
    """
    Metinleri TF-IDF vektörlerine dönüştürür.

    Önemli:
    TF-IDF bir embedding modeli değildir.
    Kelime tabanlı bir başlangıç yöntemidir.

    Daha sonra bu kısmı gerçek bir Türkçe/multilingual
    embedding modeliyle değiştireceğiz.
    """

    texts = [article["text"] for article in ARTICLES]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
    )

    vectors = vectorizer.fit_transform(texts)

    return vectorizer, vectors


def show_similarity(vectors) -> None:
    """Bir makalenin kendisi dışındaki en benzer makaleyi bulur."""

    print("\n" + "=" * 70)
    print("2. TEXT SIMILARITY")
    print("=" * 70)

    similarity_matrix = cosine_similarity(vectors)

    selected_index = 3  # A04 makalesi
    selected_article = ARTICLES[selected_index]

    similarities = similarity_matrix[selected_index].copy()

    # Bir makalenin kendisine benzerliği her zaman 1 olur.
    # Onu sonuçlardan çıkartıyoruz.
    similarities[selected_index] = -1

    most_similar_index = similarities.argmax()
    most_similar_article = ARTICLES[most_similar_index]
    score = similarities[most_similar_index]

    print(f"\nSeçilen makale: {selected_article['id']}")
    print(selected_article["text"])

    print(f"\nEn benzer bulunan makale: {most_similar_article['id']}")
    print(most_similar_article["text"])

    print(f"\nBenzerlik puanı: {score:.4f}")

    print(
        "\nBugün hesaplanan değer kelime benzerliğine dayanıyor. "
        "İleride embedding kullandığımızda anlam benzerliğine yaklaşacağız."
    )


def run_retrieval(vectorizer, vectors) -> None:
    """Bir sorguya en yakın makaleleri sıralar."""

    print("\n" + "=" * 70)
    print("3. RETRIEVAL / ARAMA")
    print("=" * 70)

    query = "Kuantum fiziğinde matematiksel denklemler"

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, vectors)[0]

    ranked_indices = scores.argsort()[::-1]

    print(f"\nKullanıcı sorgusu: {query}")
    print("\nEn ilgili ilk 5 makale:")

    for rank, article_index in enumerate(ranked_indices[:5], start=1):
        article = ARTICLES[article_index]
        score = scores[article_index]

        print(
            f"{rank}. {article['id']} "
            f"| skor={score:.4f} "
            f"| gerçek etiket={article['labels']}"
        )
        print(f"   {article['text']}")


def run_clustering(vectors):
    """Makaleleri önceden etiketleri kullanmadan kümelere ayırır."""

    print("\n" + "=" * 70)
    print("4. CLUSTERING")
    print("=" * 70)

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10,
    )

    cluster_ids = model.fit_predict(vectors)

    for cluster_id in sorted(set(cluster_ids)):
        print(f"\nCluster {cluster_id}:")

        for article, assigned_cluster in zip(ARTICLES, cluster_ids):
            if assigned_cluster == cluster_id:
                print(
                    f"- {article['id']} "
                    f"| gerçek etiket={article['labels']}"
                )

    print(
        "\nÖnemli: KMeans gerçek etiketleri kullanmadı. "
        "Yalnızca metinlerin vektörlerine bakarak gruplar oluşturdu."
    )

    return model, cluster_ids


def show_cluster_topics(model, vectorizer) -> None:
    """
    Her cluster merkezindeki en güçlü kelimeleri gösterir.

    Bu işlem basit bir topic yorumlama örneğidir.
    """

    print("\n" + "=" * 70)
    print("5. TOPIC MODELING'E BASİT BİR YAKLAŞIM")
    print("=" * 70)

    feature_names = vectorizer.get_feature_names_out()

    for cluster_id, cluster_center in enumerate(model.cluster_centers_):
        top_term_indices = cluster_center.argsort()[-6:][::-1]
        top_terms = [feature_names[index] for index in top_term_indices]

        print(f"\nCluster {cluster_id} için öne çıkan ifadeler:")
        print(", ".join(top_terms))

    print(
        "\nBu kelimelere ve temsilci makalelere bakarak "
        "clusterlara daha sonra insan tarafından konu adı verilebilir."
    )


def create_visualization(vectors, cluster_ids) -> None:
    """Yüksek boyutlu vektörleri PCA ile 2 boyuta indirip gösterir."""

    print("\n" + "=" * 70)
    print("6. BOYUT KÜÇÜLTME VE GÖRSELLEŞTİRME")
    print("=" * 70)

    dense_vectors = vectors.toarray()

    pca = PCA(
        n_components=2,
        random_state=42,
    )

    points_2d = pca.fit_transform(dense_vectors)

    project_root = Path(__file__).resolve().parents[3]
    output_directory = project_root / "research" / "outputs"
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / "day01_clusters_pca.png"

    plt.figure(figsize=(11, 7))

    plt.scatter(
        points_2d[:, 0],
        points_2d[:, 1],
        c=cluster_ids,
        s=100,
    )

    for index, article in enumerate(ARTICLES):
        plt.annotate(
            article["id"],
            (points_2d[index, 0], points_2d[index, 1]),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.title("Örnek Makalelerin PCA ile 2 Boyutlu Gösterimi")
    plt.xlabel("PCA Boyutu 1")
    plt.ylabel("PCA Boyutu 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nGörsel oluşturuldu:\n{output_path}")

    print(
        "\nNoktaların birbirine yakın görünmesi, "
        "vektörlerinin 2 boyutlu gösterimde yakın olduğu anlamına gelir."
    )


def main() -> None:
    show_classification_examples()

    vectorizer, vectors = create_vectors()

    print("\n" + "=" * 70)
    print("VEKTÖR BİLGİSİ")
    print("=" * 70)
    print(f"Makale sayısı     : {vectors.shape[0]}")
    print(f"Vektör boyutu     : {vectors.shape[1]}")
    print(f"Vektör matris şekli: {vectors.shape}")

    show_similarity(vectors)
    run_retrieval(vectorizer, vectors)

    clustering_model, cluster_ids = run_clustering(vectors)

    show_cluster_topics(clustering_model, vectorizer)
    create_visualization(vectors, cluster_ids)

    print("\n" + "=" * 70)
    print("DENEY TAMAMLANDI")
    print("=" * 70)


if __name__ == "__main__":
    main()



'''
=======================OUTPUT===============================================
1. CLASSIFICATION VE MULTI-LABEL CLASSIFICATION
======================================================================

Classification örneği:
Input : Diferansiyel denklemlerin sayısal yöntemlerle çözümü ve çözüm hatalarının matematiksel analizi incelenmiştir.
Output: Matematik

Multi-label classification örneği:
Input : Kuantum parçacıklarının davranışı dalga denklemleri ve matematiksel operatörler kullanılarak incelenmiştir.
Output: ['Fizik', 'Matematik']

Not: Bugün classification modeli eğitmiyoruz. Yalnızca bu problemin beklediği output biçimini görüyoruz.

======================================================================
VEKTÖR BİLGİSİ
======================================================================
Makale sayısı     : 12
Vektör boyutu     : 211
Vektör matris şekli: (12, 211)

======================================================================
2. TEXT SIMILARITY
======================================================================

Seçilen makale: A04
Kuantum parçacıklarının davranışı dalga denklemleri ve matematiksel operatörler kullanılarak incelenmiştir.

En benzer bulunan makale: A09
Graf sinir ağları ve matematiksel çizge modelleri kullanılarak bağlantı tahmini yapılmıştır.

Benzerlik puanı: 0.1129

Bugün hesaplanan değer kelime benzerliğine dayanıyor. İleride embedding kullandığımızda anlam benzerliğine yaklaşacağız.

======================================================================
3. RETRIEVAL / ARAMA
======================================================================

Kullanıcı sorgusu: Kuantum fiziğinde matematiksel denklemler

En ilgili ilk 5 makale:
1. A04 | skor=0.2290 | gerçek etiket=['Fizik', 'Matematik']
   Kuantum parçacıklarının davranışı dalga denklemleri ve matematiksel operatörler kullanılarak incelenmiştir.
2. A05 | skor=0.1472 | gerçek etiket=['Fizik', 'Matematik']
   Genel görelilik teorisindeki uzay zaman geometrisi diferansiyel denklemler yardımıyla modellenmiştir.
3. A09 | skor=0.0678 | gerçek etiket=['Bilgisayar Bilimleri', 'Matematik']
   Graf sinir ağları ve matematiksel çizge modelleri kullanılarak bağlantı tahmini yapılmıştır.
4. A01 | skor=0.0678 | gerçek etiket=['Matematik']
   Diferansiyel denklemlerin sayısal yöntemlerle çözümü ve çözüm hatalarının matematiksel analizi incelenmiştir.
5. A03 | skor=0.0642 | gerçek etiket=['Matematik']
   Olasılık dağılımları ve istatistiksel tahmin yöntemleri üzerine yeni bir matematiksel model önerilmiştir.

======================================================================
4. CLUSTERING
======================================================================

Cluster 0:
- A02 | gerçek etiket=['Matematik']
- A03 | gerçek etiket=['Matematik']
- A06 | gerçek etiket=['Fizik']
- A08 | gerçek etiket=['Bilgisayar Bilimleri']
- A11 | gerçek etiket=['Tıp']

Cluster 1:
- A04 | gerçek etiket=['Fizik', 'Matematik']
- A07 | gerçek etiket=['Bilgisayar Bilimleri']
- A09 | gerçek etiket=['Bilgisayar Bilimleri', 'Matematik']

Cluster 2:
- A10 | gerçek etiket=['Tıp']
- A12 | gerçek etiket=['Tıp']

Cluster 3:
- A01 | gerçek etiket=['Matematik']
- A05 | gerçek etiket=['Fizik', 'Matematik']

Önemli: KMeans gerçek etiketleri kullanmadı. Yalnızca metinlerin vektörlerine bakarak gruplar oluşturdu.

======================================================================
5. TOPIC MODELING'E BASİT BİR YAKLAŞIM
======================================================================

Cluster 0 için öne çıkan ifadeler:
yöntemleri, ve, karşılaştırılmıştır, yöntemleri karşılaştırılmıştır, için, matris ayrışımları

Cluster 1 için öne çıkan ifadeler:
kullanılarak, ve matematiksel, matematiksel, kuantum, operatörler kullanılarak, operatörler

Cluster 2 için öne çıkan ifadeler:
klinik, tedavi, klinik sonuçları, psikolojik, uygulanan psikolojik, uygulanan

Cluster 3 için öne çıkan ifadeler:
diferansiyel, denklemler yardımıyla, yardımıyla modellenmiştir, geometrisi, genel görelilik, genel

Bu kelimelere ve temsilci makalelere bakarak clusterlara daha sonra insan tarafından konu adı verilebilir.

======================================================================
6. BOYUT KÜÇÜLTME VE GÖRSELLEŞTİRME
======================================================================

Görsel oluşturuldu:
/home/goksu/trdizin-semantic-lab/research/outputs/day01_clusters_pca.png

Noktaların birbirine yakın görünmesi, vektörlerinin 2 boyutlu gösterimde yakın olduğu anlamına gelir.

======================================================================
DENEY TAMAMLANDI
======================================================================

===========================AMACIMIZ===========================================
Bu deneyde 12 örnek makale metnini TF-IDF kullanarak vektörleştirdik.
KMeans, gerçek konu etiketlerini kullanmadan bu vektörleri dört kümeye ayırdı.
Ardından yüksek boyutlu TF-IDF vektörlerini PCA ile iki boyuta indirerek görselleştirdik. 
Noktalar makaleleri, renkler KMeans clusterlarını gösteriyor. 
Clusterlar gerçek subjectlerle tam örtüşmedi; çünkü TF-IDF kelime tabanlı, veri seti çok küçük
ve KMeans’e küme sayısını biz verdik. 
Ayrıca PCA yalnızca yaklaşık bir 2D gösterim sağladığı için grafikteki uzaklıkları kesin semantik 
mesafe olarak yorumlamıyoruz.
Bu deney bize pipeline’ın mantığını ve gerçek embedding modellerine neden ihtiyaç duyduğumuzu gösterdi.

'''