from typing import Dict, List

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# 1. Kullanacağımız hazır embedding modeli
# ---------------------------------------------------------
#
# Bu model nihai model seçimimiz değildir.
# Şimdilik embedding mantığını öğrenmek için kullanıyoruz.
#

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# ---------------------------------------------------------
# 2. Küçük örnek dokümanlarımız
# ---------------------------------------------------------
#
# Bazı metinlerde aynı anlam farklı kelimelerle anlatılıyor.
# Böylece modelin yalnızca kelime eşleşmesine mi,
# yoksa anlam benzerliğine mi baktığını inceleyeceğiz.
#

DOCUMENTS: List[Dict[str, str]] = [
    {
        "id": "D01",
        "text": (
            "Kalp hastalıklarının tanısında klinik belirtiler "
            "ve tıbbi bulgular incelenmiştir."
        ),
    },
    {
        "id": "D02",
        "text": (
            "Kardiyovasküler rahatsızlıkların teşhisinde "
            "hastaların sağlık verileri değerlendirilmiştir."
        ),
    },
    {
        "id": "D03",
        "text": (
            "Güneş panellerinin enerji üretim verimliliği "
            "farklı hava koşullarında analiz edilmiştir."
        ),
    },
    {
        "id": "D04",
        "text": (
            "Fotovoltaik sistemlerde elektrik üretimi ve "
            "güneş hücrelerinin performansı araştırılmıştır."
        ),
    },
    {
        "id": "D05",
        "text": (
            "Türkçe metinler üzerinde doğal dil işleme ve "
            "metin sınıflandırma yöntemleri karşılaştırılmıştır."
        ),
    },
    {
        "id": "D06",
        "text": (
            "Mutfakta kullanılan pişirme teknikleri ve "
            "yemek hazırlama süreçleri incelenmiştir."
        ),
    },
]


def load_embedding_model() -> SentenceTransformer:
    """
    Hazır embedding modelini yükler.

    Program ilk kez çalıştırıldığında model internetten indirilir.
    Daha sonraki çalıştırmalarda önbellekteki model kullanılabilir.
    """

    print("=" * 70)
    print("1. EMBEDDING MODELİNİ YÜKLEME")
    print("=" * 70)

    print(f"\nModel adı: {MODEL_NAME}")
    print("\nModel yükleniyor...")

    model = SentenceTransformer(MODEL_NAME)

    print("Model başarıyla yüklendi.")
    print(f"Çalışılan cihaz: {model.device}")
    print(
        "Embedding boyutu: "
        f"{model.get_sentence_embedding_dimension()}"
    )

    return model


def create_embeddings(
    model: SentenceTransformer,
):
    """
    Bütün dokümanları embedding vektörlerine dönüştürür.
    """

    print("\n" + "=" * 70)
    print("2. METİNLERİ VEKTÖRE DÖNÜŞTÜRME")
    print("=" * 70)

    texts = [document["text"] for document in DOCUMENTS]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(f"\nDoküman sayısı: {len(DOCUMENTS)}")
    print(f"Vektör matrisinin şekli: {embeddings.shape}")
    print(f"Veri tipi: {embeddings.dtype}")

    print("\nD01 vektörünün ilk 8 sayısı:")

    for value in embeddings[0][:8]:
        print(f"{value:.6f}")

    print(
        "\nNot: Bu sayıların her birinin tek başına "
        "'tıp', 'fizik' veya 'matematik' gibi "
        "doğrudan bir anlamı yoktur."
    )

    return embeddings


def show_selected_similarities(embeddings) -> None:
    """
    Önceden seçilmiş bazı doküman çiftlerinin
    cosine similarity değerlerini gösterir.
    """

    print("\n" + "=" * 70)
    print("3. SEÇİLMİŞ METİN ÇİFTLERİNİN BENZERLİĞİ")
    print("=" * 70)

    similarity_matrix = cosine_similarity(embeddings)

    selected_pairs = [
        ("D01", "D02"),
        ("D01", "D03"),
        ("D03", "D04"),
        ("D05", "D06"),
    ]

    document_index = {
        document["id"]: index
        for index, document in enumerate(DOCUMENTS)
    }

    for first_id, second_id in selected_pairs:
        first_index = document_index[first_id]
        second_index = document_index[second_id]

        score = similarity_matrix[first_index][second_index]

        print(
            f"\n{first_id} ↔ {second_id}"
            f"\nCosine similarity: {score:.4f}"
        )

        print(f"{first_id}: {DOCUMENTS[first_index]['text']}")
        print(f"{second_id}: {DOCUMENTS[second_index]['text']}")


def run_semantic_search(
    model: SentenceTransformer,
    document_embeddings,
) -> None:
    """
    Kullanıcının sorgusunu embedding'e dönüştürür
    ve en yakın dokümanları sıralar.
    """

    print("\n" + "=" * 70)
    print("4. BASİT SEMANTIC SEARCH")
    print("=" * 70)

    queries = [
        "Kalp rahatsızlığını belirleme yöntemleri",
        "Güneş enerjisinden elektrik üretimi",
    ]

    for query in queries:
        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        scores = cosine_similarity(
            query_embedding,
            document_embeddings,
        )[0]

        ranked_indices = scores.argsort()[::-1]

        print(f"\nSorgu: {query}")
        print("-" * 70)

        for rank, document_index in enumerate(
            ranked_indices[:3],
            start=1,
        ):
            document = DOCUMENTS[document_index]
            score = scores[document_index]

            print(
                f"{rank}. {document['id']} "
                f"| benzerlik={score:.4f}"
            )
            print(f"   {document['text']}")


def main() -> None:
    model = load_embedding_model()

    document_embeddings = create_embeddings(model)

    show_selected_similarities(document_embeddings)

    run_semantic_search(
        model,
        document_embeddings,
    )

    print("\n" + "=" * 70)
    print("DAY 02 İLK EMBEDDING DENEYİ TAMAMLANDI")
    print("=" * 70)


if __name__ == "__main__":
    main()


'''
Model adı: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Çalışılan cihaz: cuda:0
Embedding boyutu: 384

======================================================================
2. METİNLERİ VEKTÖRE DÖNÜŞTÜRME
======================================================================
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  7.93it/s]

Doküman sayısı: 6
Vektör matrisinin şekli: (6, 384)
Veri tipi: float32

D01 vektörünün ilk 8 sayısı:
-0.015135
0.034961
-0.026734
0.085876
0.008382
0.020333
-0.043974
0.028135

Not: Bu sayıların her birinin tek başına 'tıp', 'fizik' veya 'matematik' gibi doğrudan bir anlamı yoktur.

======================================================================
3. SEÇİLMİŞ METİN ÇİFTLERİNİN BENZERLİĞİ
======================================================================

D01 ↔ D02
Cosine similarity: 0.8705
D01: Kalp hastalıklarının tanısında klinik belirtiler ve tıbbi bulgular incelenmiştir.
D02: Kardiyovasküler rahatsızlıkların teşhisinde hastaların sağlık verileri değerlendirilmiştir.

D01 ↔ D03
Cosine similarity: 0.0113
D01: Kalp hastalıklarının tanısında klinik belirtiler ve tıbbi bulgular incelenmiştir.
D03: Güneş panellerinin enerji üretim verimliliği farklı hava koşullarında analiz edilmiştir.

D03 ↔ D04
Cosine similarity: 0.7883
D03: Güneş panellerinin enerji üretim verimliliği farklı hava koşullarında analiz edilmiştir.
D04: Fotovoltaik sistemlerde elektrik üretimi ve güneş hücrelerinin performansı araştırılmıştır.

D05 ↔ D06
Cosine similarity: 0.2703
D05: Türkçe metinler üzerinde doğal dil işleme ve metin sınıflandırma yöntemleri karşılaştırılmıştır.
D06: Mutfakta kullanılan pişirme teknikleri ve yemek hazırlama süreçleri incelenmiştir.

======================================================================
4. BASİT SEMANTIC SEARCH
======================================================================

Sorgu: Kalp rahatsızlığını belirleme yöntemleri
----------------------------------------------------------------------
1. D01 | benzerlik=0.8584
   Kalp hastalıklarının tanısında klinik belirtiler ve tıbbi bulgular incelenmiştir.
2. D02 | benzerlik=0.7733
   Kardiyovasküler rahatsızlıkların teşhisinde hastaların sağlık verileri değerlendirilmiştir.
3. D05 | benzerlik=0.0586
   Türkçe metinler üzerinde doğal dil işleme ve metin sınıflandırma yöntemleri karşılaştırılmıştır.

Sorgu: Güneş enerjisinden elektrik üretimi
----------------------------------------------------------------------
1. D04 | benzerlik=0.7938
   Fotovoltaik sistemlerde elektrik üretimi ve güneş hücrelerinin performansı araştırılmıştır.
2. D03 | benzerlik=0.7563
   Güneş panellerinin enerji üretim verimliliği farklı hava koşullarında analiz edilmiştir.
3. D06 | benzerlik=0.1134
   Mutfakta kullanılan pişirme teknikleri ve yemek hazırlama süreçleri incelenmiştir.
'''