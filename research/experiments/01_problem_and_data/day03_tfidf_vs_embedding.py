import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# 1. Kullanacağımız hazır embedding modeli
# ---------------------------------------------------------

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# ---------------------------------------------------------
# 2. Gerçek konular
# ---------------------------------------------------------
#
# Bunlar modelin clustering sırasında keşfettiği konular değildir.
# Deney sonucunu yorumlayabilmek için bizim bildiğimiz etiketlerdir.
#
# PCA görsellerinde yalnızca noktaları renklendirmek amacıyla
# kullanılacaklar. TF-IDF ve embedding oluşturulurken kullanılmazlar.
#

TOPICS = [
    "Sağlık",
    "Enerji",
    "Bilgisayar Bilimleri",
    "Gastronomi",
]


# ---------------------------------------------------------
# 3. Küçük deney veri seti
# ---------------------------------------------------------

DOCUMENTS: List[Dict[str, str]] = [
    {
        "id": "D01",
        "topic": "Sağlık",
        "text": (
            "Kalp hastalıklarının tanısında klinik belirtiler "
            "ve tıbbi bulgular incelenmiştir."
        ),
    },
    {
        "id": "D02",
        "topic": "Sağlık",
        "text": (
            "Kardiyovasküler rahatsızlıkların teşhisinde "
            "hastaların sağlık verileri değerlendirilmiştir."
        ),
    },
    {
        "id": "D03",
        "topic": "Enerji",
        "text": (
            "Güneş panellerinin enerji üretim verimliliği "
            "farklı hava koşullarında analiz edilmiştir."
        ),
    },
    {
        "id": "D04",
        "topic": "Enerji",
        "text": (
            "Fotovoltaik sistemlerde elektrik üretimi ve "
            "güneş hücrelerinin performansı araştırılmıştır."
        ),
    },
    {
        "id": "D05",
        "topic": "Bilgisayar Bilimleri",
        "text": (
            "Türkçe metinler üzerinde doğal dil işleme ve "
            "metin sınıflandırma yöntemleri karşılaştırılmıştır."
        ),
    },
    {
        "id": "D06",
        "topic": "Bilgisayar Bilimleri",
        "text": (
            "Bilgisayarların insan dilini çözümlemesi için "
            "sözcük temsilleri ve otomatik kategori tahmini "
            "yöntemleri geliştirilmiştir."
        ),
    },
    {
        "id": "D07",
        "topic": "Gastronomi",
        "text": (
            "Mutfakta kullanılan pişirme teknikleri ve "
            "yemek hazırlama süreçleri incelenmiştir."
        ),
    },
    {
        "id": "D08",
        "topic": "Gastronomi",
        "text": (
            "Gıdaların ısıl işlemle hazırlanması ve "
            "gastronomik üretim aşamaları değerlendirilmiştir."
        ),
    },
]


# ---------------------------------------------------------
# 4. Semantic search sorgularımız
# ---------------------------------------------------------
#
# Bazı sorgular dokümanlarla aynı kelimeleri içermiyor.
# Bu sayede TF-IDF ile embedding arasındaki farkı
# daha açık görebiliriz.
#

QUERIES: List[Dict[str, str]] = [
    {
        "id": "Q01",
        "text": "Kardiyak bozuklukların teşhis edilmesi",
        "expected_topic": "Sağlık",
    },
    {
        "id": "Q02",
        "text": "Işık enerjisinin elektriğe dönüştürülmesi",
        "expected_topic": "Enerji",
    },
    {
        "id": "Q03",
        "text": "Bilgisayarların insan dilini anlayıp sınıflandırması",
        "expected_topic": "Bilgisayar Bilimleri",
    },
    {
        "id": "Q04",
        "text": "Yemek pişirme ve hazırlama teknikleri",
        "expected_topic": "Gastronomi",
    },
]


# ---------------------------------------------------------
# 5. Önceden seçilmiş doküman çiftleri
# ---------------------------------------------------------

SELECTED_PAIRS: List[Tuple[str, str]] = [
    ("D01", "D02"),
    ("D03", "D04"),
    ("D05", "D06"),
    ("D07", "D08"),
    ("D01", "D03"),
]


def get_output_directory() -> Path:
    """Projenin outputs klasörünü bulur ve gerekirse oluşturur."""

    project_root = Path(__file__).resolve().parents[3]
    output_directory = project_root / "research" / "outputs"
    output_directory.mkdir(parents=True, exist_ok=True)

    return output_directory


def get_document_texts() -> List[str]:
    """Doküman metinlerini tek bir liste olarak döndürür."""

    return [document["text"] for document in DOCUMENTS]


def get_document_index() -> Dict[str, int]:
    """D01 gibi bir kimliği, listedeki sıra numarasına çevirir."""

    return {
        document["id"]: index
        for index, document in enumerate(DOCUMENTS)
    }


def create_tfidf_vectors():
    """
    Dokümanları kelime tabanlı TF-IDF vektörlerine dönüştürür.
    """

    print("=" * 75)
    print("1. TF-IDF VEKTÖRLERİNİ OLUŞTURMA")
    print("=" * 75)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
    )

    vectors = vectorizer.fit_transform(get_document_texts())

    print(f"\nDoküman sayısı : {vectors.shape[0]}")
    print(f"TF-IDF boyutu  : {vectors.shape[1]}")
    print(f"Matris şekli   : {vectors.shape}")

    return vectorizer, vectors


def create_embedding_vectors():
    """
    Dokümanları semantic embedding vektörlerine dönüştürür.
    """

    print("\n" + "=" * 75)
    print("2. SEMANTIC EMBEDDING VEKTÖRLERİNİ OLUŞTURMA")
    print("=" * 75)

    print(f"\nModel yükleniyor: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print(f"Çalışılan cihaz : {model.device}")
    print(
        "Embedding boyutu: "
        f"{model.get_sentence_embedding_dimension()}"
    )

    vectors = model.encode(
        get_document_texts(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(f"\nDoküman sayısı       : {vectors.shape[0]}")
    print(f"Embedding boyutu     : {vectors.shape[1]}")
    print(f"Embedding matris şekli: {vectors.shape}")

    return model, vectors


def compare_selected_pairs(
    tfidf_vectors,
    embedding_vectors: np.ndarray,
) -> None:
    """
    Seçilen doküman çiftlerinin iki yöntemdeki
    cosine similarity puanlarını karşılaştırır.
    """

    print("\n" + "=" * 75)
    print("3. DOKÜMAN ÇİFTLERİNİ KARŞILAŞTIRMA")
    print("=" * 75)

    document_index = get_document_index()

    tfidf_similarity_matrix = cosine_similarity(tfidf_vectors)
    embedding_similarity_matrix = cosine_similarity(
        embedding_vectors
    )

    for first_id, second_id in SELECTED_PAIRS:
        first_index = document_index[first_id]
        second_index = document_index[second_id]

        tfidf_score = tfidf_similarity_matrix[
            first_index, second_index
        ]

        embedding_score = embedding_similarity_matrix[
            first_index, second_index
        ]

        first_document = DOCUMENTS[first_index]
        second_document = DOCUMENTS[second_index]

        print("\n" + "-" * 75)
        print(f"{first_id} ↔ {second_id}")
        print(
            f"Gerçek konular: "
            f"{first_document['topic']} ↔ "
            f"{second_document['topic']}"
        )
        print(f"TF-IDF benzerliği   : {tfidf_score:.4f}")
        print(f"Embedding benzerliği: {embedding_score:.4f}")

        print(f"\n{first_id}: {first_document['text']}")
        print(f"{second_id}: {second_document['text']}")

    print(
        "\nÖnemli not: TF-IDF ve embedding farklı vektör "
        "uzaylarıdır. Bu nedenle puanları doğrudan aynı "
        "ölçekmiş gibi yorumlamamalıyız. Asıl olarak "
        "sıralamalara ve ilgili-ilgisiz ayrımına bakacağız."
    )


def get_ranked_indices(scores: np.ndarray) -> np.ndarray:
    """Skorları en yüksekten en düşüğe doğru sıralar."""

    return scores.argsort()[::-1]


def print_method_results(
    method_name: str,
    scores: np.ndarray,
    top_k: int = 3,
) -> None:
    """Bir retrieval yönteminin ilk sonuçlarını terminale yazdırır."""

    ranked_indices = get_ranked_indices(scores)

    print(f"\n{method_name} sonuçları:")

    if np.allclose(scores, 0):
        print(
            "UYARI: Bütün skorlar sıfır. Sorgu ile TF-IDF "
            "sözlüğü arasında ortak kelime bulunamamış olabilir."
        )

    for rank, document_index in enumerate(
        ranked_indices[:top_k],
        start=1,
    ):
        document = DOCUMENTS[document_index]
        score = scores[document_index]

        print(
            f"{rank}. {document['id']} "
            f"| skor={score:.4f} "
            f"| konu={document['topic']}"
        )
        print(f"   {document['text']}")


def compare_retrieval_results(
    tfidf_vectorizer: TfidfVectorizer,
    tfidf_document_vectors,
    embedding_model: SentenceTransformer,
    embedding_document_vectors: np.ndarray,
) -> None:
    """
    Aynı sorguyu TF-IDF ve embedding yöntemiyle aratır.
    Sonuçları hem terminale hem CSV dosyasına yazar.
    """

    print("\n" + "=" * 75)
    print("4. TF-IDF VE SEMANTIC SEARCH KARŞILAŞTIRMASI")
    print("=" * 75)

    output_directory = get_output_directory()
    csv_path = (
        output_directory
        / "day03_retrieval_comparison.csv"
    )

    csv_rows: List[Dict[str, object]] = []

    for query in QUERIES:
        query_id = query["id"]
        query_text = query["text"]
        expected_topic = query["expected_topic"]

        print("\n" + "=" * 75)
        print(f"{query_id}: {query_text}")
        print(f"Beklenen konu: {expected_topic}")
        print("=" * 75)

        # TF-IDF sorgu vektörü
        tfidf_query_vector = tfidf_vectorizer.transform(
            [query_text]
        )

        tfidf_scores = cosine_similarity(
            tfidf_query_vector,
            tfidf_document_vectors,
        )[0]

        # Semantic embedding sorgu vektörü
        embedding_query_vector = embedding_model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embedding_scores = cosine_similarity(
            embedding_query_vector,
            embedding_document_vectors,
        )[0]

        print_method_results(
            method_name="TF-IDF",
            scores=tfidf_scores,
        )

        print_method_results(
            method_name="SEMANTIC EMBEDDING",
            scores=embedding_scores,
        )

        for method_name, scores in [
            ("TF-IDF", tfidf_scores),
            ("Embedding", embedding_scores),
        ]:
            ranked_indices = get_ranked_indices(scores)

            for rank, document_index in enumerate(
                ranked_indices[:3],
                start=1,
            ):
                document = DOCUMENTS[document_index]

                csv_rows.append(
                    {
                        "query_id": query_id,
                        "query_text": query_text,
                        "expected_topic": expected_topic,
                        "method": method_name,
                        "rank": rank,
                        "document_id": document["id"],
                        "document_topic": document["topic"],
                        "score": float(
                            scores[document_index]
                        ),
                        "document_text": document["text"],
                    }
                )

    with csv_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        fieldnames = [
            "query_id",
            "query_text",
            "expected_topic",
            "method",
            "rank",
            "document_id",
            "document_topic",
            "score",
            "document_text",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nCSV karşılaştırma raporu oluşturuldu:\n{csv_path}")


def convert_to_dense(vectors) -> np.ndarray:
    """
    Sparse TF-IDF matrisini veya NumPy matrisini
    PCA için dense NumPy dizisine dönüştürür.
    """

    if hasattr(vectors, "toarray"):
        return vectors.toarray()

    return np.asarray(vectors)


def create_pca_visualization(
    vectors,
    output_filename: str,
    title: str,
) -> None:
    """
    Yüksek boyutlu vektörleri PCA ile 2 boyuta indirir
    ve gerçek konu etiketlerine göre renklendirir.

    Konu etiketleri PCA hesabında kullanılmaz.
    Yalnızca sonuçları yorumlamak için renk verir.
    """

    dense_vectors = convert_to_dense(vectors)

    pca = PCA(
        n_components=2,
        random_state=42,
    )

    points_2d = pca.fit_transform(dense_vectors)

    topic_ids = [
        TOPICS.index(document["topic"])
        for document in DOCUMENTS
    ]

    explained_variance = (
        pca.explained_variance_ratio_.sum() * 100
    )

    output_path = (
        get_output_directory()
        / output_filename
    )

    plt.figure(figsize=(11, 7))

    scatter = plt.scatter(
        points_2d[:, 0],
        points_2d[:, 1],
        c=topic_ids,
        s=120,
    )

    for index, document in enumerate(DOCUMENTS):
        plt.annotate(
            document["id"],
            (
                points_2d[index, 0],
                points_2d[index, 1],
            ),
            xytext=(6, 6),
            textcoords="offset points",
        )

    legend_handles, _ = scatter.legend_elements()

    plt.legend(
        legend_handles,
        TOPICS,
        title="Bilinen konu",
    )

    plt.title(
        f"{title}\n"
        f"İlk iki PCA boyutunun koruduğu varyans: "
        f"%{explained_variance:.2f}"
    )

    plt.xlabel("PCA Boyutu 1")
    plt.ylabel("PCA Boyutu 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print(f"Görsel oluşturuldu:\n{output_path}")


def main() -> None:
    tfidf_vectorizer, tfidf_vectors = (
        create_tfidf_vectors()
    )

    embedding_model, embedding_vectors = (
        create_embedding_vectors()
    )

    compare_selected_pairs(
        tfidf_vectors=tfidf_vectors,
        embedding_vectors=embedding_vectors,
    )

    compare_retrieval_results(
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_document_vectors=tfidf_vectors,
        embedding_model=embedding_model,
        embedding_document_vectors=embedding_vectors,
    )

    print("\n" + "=" * 75)
    print("5. PCA GÖRSELLERİNİ OLUŞTURMA")
    print("=" * 75)

    create_pca_visualization(
        vectors=tfidf_vectors,
        output_filename="day03_tfidf_pca.png",
        title="TF-IDF Vektörlerinin 2 Boyutlu Gösterimi",
    )

    create_pca_visualization(
        vectors=embedding_vectors,
        output_filename="day03_embedding_pca.png",
        title="Semantic Embedding Vektörlerinin 2 Boyutlu Gösterimi",
    )

    print("\n" + "=" * 75)
    print("DAY 03 KARŞILAŞTIRMA DENEYİ TAMAMLANDI")
    print("=" * 75)


if __name__ == "__main__":
    main()

'''
1. TF-IDF VEKTÖRLERİNİ OLUŞTURMA
===========================================================================

Doküman sayısı : 8
TF-IDF boyutu  : 136
Matris şekli   : (8, 136)

===========================================================================
2. SEMANTIC EMBEDDING VEKTÖRLERİNİ OLUŞTURMA
===========================================================================

Model yükleniyor: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Çalışılan cihaz : cuda:0
Embedding boyutu: 384
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  8.61it/s]

Doküman sayısı       : 8
Embedding boyutu     : 384
Embedding matris şekli: (8, 384)

===========================================================================
3. DOKÜMAN ÇİFTLERİNİ KARŞILAŞTIRMA
===========================================================================

---------------------------------------------------------------------------
D01 ↔ D02
Gerçek konular: Sağlık ↔ Sağlık
TF-IDF benzerliği   : 0.0000
Embedding benzerliği: 0.8705

D01: Kalp hastalıklarının tanısında klinik belirtiler ve tıbbi bulgular incelenmiştir.
D02: Kardiyovasküler rahatsızlıkların teşhisinde hastaların sağlık verileri değerlendirilmiştir.

---------------------------------------------------------------------------
D03 ↔ D04
Gerçek konular: Enerji ↔ Enerji
TF-IDF benzerliği   : 0.0410
Embedding benzerliği: 0.7883

D03: Güneş panellerinin enerji üretim verimliliği farklı hava koşullarında analiz edilmiştir.
D04: Fotovoltaik sistemlerde elektrik üretimi ve güneş hücrelerinin performansı araştırılmıştır.

---------------------------------------------------------------------------
D05 ↔ D06
Gerçek konular: Bilgisayar Bilimleri ↔ Bilgisayar Bilimleri
TF-IDF benzerliği   : 0.0436
Embedding benzerliği: 0.6024

D05: Türkçe metinler üzerinde doğal dil işleme ve metin sınıflandırma yöntemleri karşılaştırılmıştır.
D06: Bilgisayarların insan dilini çözümlemesi için sözcük temsilleri ve otomatik kategori tahmini yöntemleri geliştirilmiştir.

---------------------------------------------------------------------------
D07 ↔ D08
Gerçek konular: Gastronomi ↔ Gastronomi
TF-IDF benzerliği   : 0.0158
Embedding benzerliği: 0.7622

D07: Mutfakta kullanılan pişirme teknikleri ve yemek hazırlama süreçleri incelenmiştir.
D08: Gıdaların ısıl işlemle hazırlanması ve gastronomik üretim aşamaları değerlendirilmiştir.

---------------------------------------------------------------------------
D01 ↔ D03
Gerçek konular: Sağlık ↔ Enerji
TF-IDF benzerliği   : 0.0000
Embedding benzerliği: 0.0113

D01: Kalp hastalıklarının tanısında klinik belirtiler ve tıbbi bulgular incelenmiştir.
D03: Güneş panellerinin enerji üretim verimliliği farklı hava koşullarında analiz edilmiştir.

Önemli not: TF-IDF ve embedding farklı vektör uzaylarıdır. Bu nedenle puanları doğrudan aynı ölçekmiş gibi yorumlamamalıyız.
Asıl olarak sıralamalara ve ilgili-ilgisiz ayrımına bakacağız.

===========================================================================
4. TF-IDF VE SEMANTIC SEARCH KARŞILAŞTIRMASI
===========================================================================

===========================================================================
Q01: Kardiyak bozuklukların teşhis edilmesi
Beklenen konu: Sağlık
===========================================================================

TF-IDF sonuçları:
UYARI: Bütün skorlar sıfır. Sorgu ile TF-IDF sözlüğü arasında ortak kelime bulunamamış olabilir.
1. D08 | skor=0.0000 | konu=Gastronomi
   Gıdaların ısıl işlemle hazırlanması ve gastronomik üretim aşamaları değerlendirilmiştir.
2. D07 | skor=0.0000 | konu=Gastronomi
   Mutfakta kullanılan pişirme teknikleri ve yemek hazırlama süreçleri incelenmiştir.
3. D06 | skor=0.0000 | konu=Bilgisayar Bilimleri
   Bilgisayarların insan dilini çözümlemesi için sözcük temsilleri ve otomatik kategori tahmini yöntemleri geliştirilmiştir.

SEMANTIC EMBEDDING sonuçları:
1. D01 | skor=0.7313 | konu=Sağlık
   Kalp hastalıklarının tanısında klinik belirtiler ve tıbbi bulgular incelenmiştir.
2. D02 | skor=0.6998 | konu=Sağlık
   Kardiyovasküler rahatsızlıkların teşhisinde hastaların sağlık verileri değerlendirilmiştir.
3. D06 | skor=0.0602 | konu=Bilgisayar Bilimleri
   Bilgisayarların insan dilini çözümlemesi için sözcük temsilleri ve otomatik kategori tahmini yöntemleri geliştirilmiştir.

===========================================================================
Q02: Işık enerjisinin elektriğe dönüştürülmesi
Beklenen konu: Enerji
===========================================================================

TF-IDF sonuçları:
UYARI: Bütün skorlar sıfır. Sorgu ile TF-IDF sözlüğü arasında ortak kelime bulunamamış olabilir.
1. D08 | skor=0.0000 | konu=Gastronomi
   Gıdaların ısıl işlemle hazırlanması ve gastronomik üretim aşamaları değerlendirilmiştir.
2. D07 | skor=0.0000 | konu=Gastronomi
   Mutfakta kullanılan pişirme teknikleri ve yemek hazırlama süreçleri incelenmiştir.
3. D06 | skor=0.0000 | konu=Bilgisayar Bilimleri
   Bilgisayarların insan dilini çözümlemesi için sözcük temsilleri ve otomatik kategori tahmini yöntemleri geliştirilmiştir.

SEMANTIC EMBEDDING sonuçları:
1. D04 | skor=0.4702 | konu=Enerji
   Fotovoltaik sistemlerde elektrik üretimi ve güneş hücrelerinin performansı araştırılmıştır.
2. D03 | skor=0.3481 | konu=Enerji
   Güneş panellerinin enerji üretim verimliliği farklı hava koşullarında analiz edilmiştir.
3. D08 | skor=0.1331 | konu=Gastronomi
   Gıdaların ısıl işlemle hazırlanması ve gastronomik üretim aşamaları değerlendirilmiştir.

===========================================================================
Q03: Bilgisayarların insan dilini anlayıp sınıflandırması
Beklenen konu: Bilgisayar Bilimleri
===========================================================================

TF-IDF sonuçları:
1. D06 | skor=0.4569 | konu=Bilgisayar Bilimleri
   Bilgisayarların insan dilini çözümlemesi için sözcük temsilleri ve otomatik kategori tahmini yöntemleri geliştirilmiştir.
2. D08 | skor=0.0000 | konu=Gastronomi
   Gıdaların ısıl işlemle hazırlanması ve gastronomik üretim aşamaları değerlendirilmiştir.
3. D07 | skor=0.0000 | konu=Gastronomi
   Mutfakta kullanılan pişirme teknikleri ve yemek hazırlama süreçleri incelenmiştir.

SEMANTIC EMBEDDING sonuçları:
1. D06 | skor=0.8282 | konu=Bilgisayar Bilimleri
   Bilgisayarların insan dilini çözümlemesi için sözcük temsilleri ve otomatik kategori tahmini yöntemleri geliştirilmiştir.
2. D05 | skor=0.6038 | konu=Bilgisayar Bilimleri
   Türkçe metinler üzerinde doğal dil işleme ve metin sınıflandırma yöntemleri karşılaştırılmıştır.
3. D07 | skor=0.1885 | konu=Gastronomi
   Mutfakta kullanılan pişirme teknikleri ve yemek hazırlama süreçleri incelenmiştir.

===========================================================================
Q04: Yemek pişirme ve hazırlama teknikleri
Beklenen konu: Gastronomi
===========================================================================

TF-IDF sonuçları:
1. D07 | skor=0.5161 | konu=Gastronomi
   Mutfakta kullanılan pişirme teknikleri ve yemek hazırlama süreçleri incelenmiştir.
2. D08 | skor=0.0306 | konu=Gastronomi
   Gıdaların ısıl işlemle hazırlanması ve gastronomik üretim aşamaları değerlendirilmiştir.
3. D04 | skor=0.0303 | konu=Enerji
   Fotovoltaik sistemlerde elektrik üretimi ve güneş hücrelerinin performansı araştırılmıştır.

SEMANTIC EMBEDDING sonuçları:
1. D07 | skor=0.8617 | konu=Gastronomi
   Mutfakta kullanılan pişirme teknikleri ve yemek hazırlama süreçleri incelenmiştir.
2. D08 | skor=0.7305 | konu=Gastronomi
   Gıdaların ısıl işlemle hazırlanması ve gastronomik üretim aşamaları değerlendirilmiştir.
3. D03 | skor=0.2115 | konu=Enerji
   Güneş panellerinin enerji üretim verimliliği farklı hava koşullarında analiz edilmiştir.

CSV karşılaştırma raporu oluşturuldu:
/home/goksu/trdizin-semantic-lab/research/outputs/day03_retrieval_comparison.csv

===========================================================================
5. PCA GÖRSELLERİNİ OLUŞTURMA
===========================================================================
Görsel oluşturuldu:
/home/goksu/trdizin-semantic-lab/research/outputs/day03_tfidf_pca.png
Görsel oluşturuldu:
/home/goksu/trdizin-semantic-lab/research/outputs/day03_embedding_pca.png

===========================================================================
DAY 03 KARŞILAŞTIRMA DENEYİ TAMAMLANDI
===========================================================================
'''