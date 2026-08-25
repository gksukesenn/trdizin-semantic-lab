import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------
# 1. Deney ayarları
# ---------------------------------------------------------

TOP_K = 5
ANCHOR_ARTICLE_COUNT = 12
RANDOM_SEED = 42


# Day 10 aşamasında oluşturduğumuz embedding dosyaları.
MODEL_FILES = {
    "MiniLM": "minilm.npy",
    "TR-MTEB": "tr-mteb.npy",
    "E5-large": "e5-large.npy",
    "GTE-multilingual": "gte-multilingual.npy",
}


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


def load_articles() -> List[Dict[str, Any]]:
    """
    Day 10 benchmark deneyinde kullanılan 200 makaleyi okur.

    Embedding satırlarının sırası bu dosyadaki
    makale sırasıyla aynıdır.
    """

    input_path = (
        get_project_root()
        / "data"
        / "processed"
        / "benchmark_articles_200.jsonl"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Benchmark makale dosyası bulunamadı:\n{input_path}"
        )

    articles: List[Dict[str, Any]] = []

    with input_path.open(
        mode="r",
        encoding="utf-8",
    ) as input_file:
        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            cleaned_line = line.strip()

            if not cleaned_line:
                continue

            try:
                article = json.loads(cleaned_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL satırı okunamadı: {line_number}"
                ) from error

            if not isinstance(article, dict):
                continue

            articles.append(article)

    return articles


def normalize_embedding_rows(
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    Vektörleri güvenli biçimde yeniden normalize eder.

    Day 10'da zaten normalize ettik. Bu kontrol,
    dot product ile cosine similarity hesaplayabilmemizi
    garanti altına alır.
    """

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    non_zero_norms = np.where(
        norms == 0,
        1,
        norms,
    )

    return embeddings / non_zero_norms


def load_model_embeddings(
    article_count: int,
) -> Dict[str, np.ndarray]:
    """Bütün modellerin embedding matrislerini yükler."""

    embedding_directory = (
        get_project_root()
        / "research" / "outputs"
        / "day10_embeddings"
    )

    model_embeddings: Dict[str, np.ndarray] = {}

    for model_name, filename in MODEL_FILES.items():
        embedding_path = (
            embedding_directory
            / filename
        )

        if not embedding_path.exists():
            raise FileNotFoundError(
                f"{model_name} embedding dosyası bulunamadı:\n"
                f"{embedding_path}"
            )

        embeddings = np.load(
            embedding_path,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                f"{model_name} embedding matrisi iki boyutlu değil: "
                f"{embeddings.shape}"
            )

        if embeddings.shape[0] != article_count:
            raise ValueError(
                f"{model_name} satır sayısı ile makale sayısı "
                f"uyuşmuyor.\n"
                f"Embedding satırı: {embeddings.shape[0]}\n"
                f"Makale sayısı   : {article_count}"
            )

        normalized_embeddings = normalize_embedding_rows(
            embeddings.astype(
                np.float32,
                copy=False,
            )
        )

        model_embeddings[
            model_name
        ] = normalized_embeddings

        print(
            f"{model_name:18} "
            f"→ şekil={normalized_embeddings.shape}"
        )

    return model_embeddings


def select_anchor_indices(
    article_count: int,
) -> List[int]:
    """
    İnsan incelemesi için sabit seed ile
    12 makale seçer.

    Aynı kod yeniden çalıştırıldığında
    aynı makaleler seçilir.
    """

    if article_count <= ANCHOR_ARTICLE_COUNT:
        return list(
            range(article_count)
        )

    random_generator = random.Random(
        RANDOM_SEED
    )

    selected_indices = random_generator.sample(
        range(article_count),
        ANCHOR_ARTICLE_COUNT,
    )

    return sorted(
        selected_indices
    )


def find_nearest_neighbors(
    embeddings: np.ndarray,
    anchor_index: int,
    top_k: int,
) -> List[Tuple[int, float]]:
    """
    Bir makalenin en yakın komşularını bulur.

    Vektörler normalize olduğu için:
    dot product = cosine similarity
    """

    anchor_vector = embeddings[
        anchor_index
    ]

    similarity_scores = (
        embeddings
        @ anchor_vector
    )

    # Makalenin kendisi her zaman en yüksek skoru alır.
    # Kendisini komşu sonuçlarından çıkartıyoruz.
    similarity_scores[
        anchor_index
    ] = -np.inf

    ranked_indices = np.argsort(
        similarity_scores
    )[::-1]

    results: List[Tuple[int, float]] = []

    for neighbor_index in ranked_indices[:top_k]:
        results.append(
            (
                int(neighbor_index),
                float(
                    similarity_scores[
                        neighbor_index
                    ]
                ),
            )
        )

    return results


def normalize_keywords(value: Any) -> List[str]:
    """Keywords değerini string listesine dönüştürür."""

    if not isinstance(value, list):
        return []

    return [
        str(keyword).strip()
        for keyword in value
        if str(keyword).strip()
    ]


def keywords_as_text(value: Any) -> str:
    """Keywords listesini okunabilir metne dönüştürür."""

    keywords = normalize_keywords(value)

    if not keywords:
        return "-"

    return ", ".join(keywords)


def markdown_escape(value: Any) -> str:
    """Markdown tablo karakterlerini temizler."""

    text = str(value or "")

    return (
        text
        .replace("|", "\\|")
        .replace("\n", " ")
    )


def build_review_rows(
    articles: List[Dict[str, Any]],
    model_embeddings: Dict[str, np.ndarray],
    anchor_indices: List[int],
) -> List[Dict[str, Any]]:
    """
    İnsan değerlendirme CSV'sinin satırlarını oluşturur.
    """

    review_rows: List[Dict[str, Any]] = []

    for anchor_number, anchor_index in enumerate(
        anchor_indices,
        start=1,
    ):
        anchor_article = articles[
            anchor_index
        ]

        for model_name, embeddings in (
            model_embeddings.items()
        ):
            neighbors = find_nearest_neighbors(
                embeddings=embeddings,
                anchor_index=anchor_index,
                top_k=TOP_K,
            )

            for rank, (
                neighbor_index,
                similarity_score,
            ) in enumerate(
                neighbors,
                start=1,
            ):
                neighbor_article = articles[
                    neighbor_index
                ]

                review_rows.append(
                    {
                        "anchor_number": anchor_number,
                        "anchor_index": anchor_index,
                        "anchor_article_id": (
                            anchor_article.get(
                                "article_id",
                                "",
                            )
                        ),
                        "anchor_year": (
                            anchor_article.get(
                                "publication_year",
                                "",
                            )
                        ),
                        "anchor_title": (
                            anchor_article.get(
                                "title_tr",
                                "",
                            )
                        ),
                        "anchor_keywords": (
                            keywords_as_text(
                                anchor_article.get(
                                    "keywords_tr"
                                )
                            )
                        ),
                        "model_name": model_name,
                        "rank": rank,
                        "neighbor_article_id": (
                            neighbor_article.get(
                                "article_id",
                                "",
                            )
                        ),
                        "neighbor_year": (
                            neighbor_article.get(
                                "publication_year",
                                "",
                            )
                        ),
                        "neighbor_title": (
                            neighbor_article.get(
                                "title_tr",
                                "",
                            )
                        ),
                        "neighbor_keywords": (
                            keywords_as_text(
                                neighbor_article.get(
                                    "keywords_tr"
                                )
                            )
                        ),
                        "similarity_score": (
                            similarity_score
                        ),

                        # Bu iki sütunu daha sonra elle dolduracağız.
                        #
                        # Önerilen değerler:
                        # ilgili
                        # kısmen ilgili
                        # ilgisiz
                        "human_judgment": "",
                        "reviewer_note": "",
                    }
                )

    return review_rows


def save_review_csv(
    review_rows: List[Dict[str, Any]],
) -> Path:
    """
    İnsan değerlendirmesi için CSV dosyası oluşturur.
    """

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day11_semantic_neighbor_review.csv"
    )

    fieldnames = [
        "anchor_number",
        "anchor_index",
        "anchor_article_id",
        "anchor_year",
        "anchor_title",
        "anchor_keywords",
        "model_name",
        "rank",
        "neighbor_article_id",
        "neighbor_year",
        "neighbor_title",
        "neighbor_keywords",
        "similarity_score",
        "human_judgment",
        "reviewer_note",
    ]

    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            review_rows
        )

    return output_path


def save_markdown_report(
    articles: List[Dict[str, Any]],
    model_embeddings: Dict[str, np.ndarray],
    anchor_indices: List[int],
) -> Path:
    """
    Sonuçları VSCode içinde kolay okunabilen
    Markdown raporu olarak kaydeder.
    """

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day11_semantic_neighbor_review.md"
    )

    lines: List[str] = [
        "# Semantic Komşuluk Model Karşılaştırması",
        "",
        (
            "Bu raporda her seçili makale için dört embedding "
            "modelinin en yakın beş makalesi gösterilmektedir."
        ),
        "",
        (
            "Bu aşamada subject etiketleri kullanılmamıştır. "
            "İnceleme başlık ve anahtar kelimeler üzerinden yapılır."
        ),
        "",
    ]

    for anchor_number, anchor_index in enumerate(
        anchor_indices,
        start=1,
    ):
        anchor_article = articles[
            anchor_index
        ]

        lines.extend(
            [
                "---",
                "",
                (
                    f"## Anchor {anchor_number}: "
                    f"{markdown_escape(anchor_article.get('title_tr'))}"
                ),
                "",
                (
                    f"**Makale ID:** "
                    f"{markdown_escape(anchor_article.get('article_id'))}"
                ),
                "",
                (
                    f"**Yıl:** "
                    f"{markdown_escape(anchor_article.get('publication_year'))}"
                ),
                "",
                (
                    f"**Anahtar kelimeler:** "
                    f"{markdown_escape(keywords_as_text(anchor_article.get('keywords_tr')))}"
                ),
                "",
                (
                    f"**Özet başlangıcı:** "
                    f"{markdown_escape(anchor_article.get('abstract_tr', '')[:400])}..."
                ),
                "",
            ]
        )

        for model_name, embeddings in (
            model_embeddings.items()
        ):
            neighbors = find_nearest_neighbors(
                embeddings=embeddings,
                anchor_index=anchor_index,
                top_k=TOP_K,
            )

            lines.extend(
                [
                    f"### {model_name}",
                    "",
                    (
                        "| Sıra | Skor | Makale | "
                        "Anahtar kelimeler |"
                    ),
                    (
                        "|---:|---:|---|---|"
                    ),
                ]
            )

            for rank, (
                neighbor_index,
                similarity_score,
            ) in enumerate(
                neighbors,
                start=1,
            ):
                neighbor_article = articles[
                    neighbor_index
                ]

                title = markdown_escape(
                    neighbor_article.get(
                        "title_tr",
                        "",
                    )
                )

                keywords = markdown_escape(
                    keywords_as_text(
                        neighbor_article.get(
                            "keywords_tr"
                        )
                    )
                )

                lines.append(
                    f"| {rank} "
                    f"| {similarity_score:.4f} "
                    f"| {title} "
                    f"| {keywords} |"
                )

            lines.append("")

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as markdown_file:
        markdown_file.write(
            "\n".join(lines)
        )

    return output_path


def print_selected_anchors(
    articles: List[Dict[str, Any]],
    anchor_indices: List[int],
) -> None:
    """Seçilen anchor makaleleri terminalde gösterir."""

    print("\n" + "=" * 75)
    print("İNCELEME İÇİN SEÇİLEN MAKALELER")
    print("=" * 75)

    for anchor_number, anchor_index in enumerate(
        anchor_indices,
        start=1,
    ):
        article = articles[
            anchor_index
        ]

        print(
            f"\n{anchor_number:2}. "
            f"ID={article.get('article_id')} "
            f"| yıl={article.get('publication_year')}"
        )

        print(
            f"    {article.get('title_tr')}"
        )


def main() -> None:
    print("=" * 75)
    print("SEMANTIC KOMŞULUK KARŞILAŞTIRMASI")
    print("=" * 75)

    articles = load_articles()

    print(
        f"\nOkunan benchmark makalesi: "
        f"{len(articles)}"
    )

    print("\nEmbedding dosyaları yükleniyor:")

    model_embeddings = load_model_embeddings(
        article_count=len(articles)
    )

    anchor_indices = select_anchor_indices(
        article_count=len(articles)
    )

    print_selected_anchors(
        articles=articles,
        anchor_indices=anchor_indices,
    )

    review_rows = build_review_rows(
        articles=articles,
        model_embeddings=model_embeddings,
        anchor_indices=anchor_indices,
    )

    csv_path = save_review_csv(
        review_rows
    )

    markdown_path = save_markdown_report(
        articles=articles,
        model_embeddings=model_embeddings,
        anchor_indices=anchor_indices,
    )

    print("\n" + "=" * 75)
    print("KARŞILAŞTIRMA DOSYALARI OLUŞTURULDU")
    print("=" * 75)

    print(
        f"\nİnsan değerlendirme CSV:\n"
        f"{csv_path}"
    )

    print(
        f"\nOkunabilir Markdown raporu:\n"
        f"{markdown_path}"
    )

    print(
        "\nCSV içindeki human_judgment sütununu "
        "'ilgili', 'kısmen ilgili' veya 'ilgisiz' "
        "olarak doldurabiliriz."
    )


if __name__ == "__main__":
    main()