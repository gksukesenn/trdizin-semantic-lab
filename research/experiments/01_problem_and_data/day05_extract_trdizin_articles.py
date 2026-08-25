import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


def load_raw_response() -> Dict[str, Any]:
    """Day 04 aşamasında kaydettiğimiz ham JSON'u okur."""

    input_path = (
        get_project_root()
        / "data"
        / "raw"
        / "day04_trdizin_search_response.json"
    )

    with input_path.open(
        mode="r",
        encoding="utf-8",
    ) as json_file:
        return json.load(json_file)


def find_turkish_abstract(
    abstract_items: Any,
) -> Optional[Dict[str, Any]]:
    """
    abstracts listesinden language değeri TUR olan kaydı bulur.

    Makale Türkçe yayın olsa bile API aynı makalenin
    Türkçe ve İngilizce abstractını birlikte döndürebilir.
    """

    if not isinstance(abstract_items, list):
        return None

    for abstract_item in abstract_items:
        if not isinstance(abstract_item, dict):
            continue

        if abstract_item.get("language") == "TUR":
            return abstract_item

    return None


def normalize_keywords(value: Any) -> List[str]:
    """Keywords alanını her durumda liste biçimine dönüştürür."""

    if isinstance(value, list):
        return [
            str(keyword).strip()
            for keyword in value
            if str(keyword).strip()
        ]

    if isinstance(value, str) and value.strip():
        return [value.strip()]

    return []


def extract_articles(
    data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """TR Dizin hit kayıtlarını sade makale kayıtlarına dönüştürür."""

    hits = (
        data
        .get("hits", {})
        .get("hits", [])
    )

    extracted_articles: List[Dict[str, Any]] = []

    for hit in hits:
        if not isinstance(hit, dict):
            continue

        source = hit.get("_source", {})

        if not isinstance(source, dict):
            continue

        turkish_item = find_turkish_abstract(
            source.get("abstracts")
        )

        if turkish_item is None:
            title = None
            abstract = None
            keywords: List[str] = []
        else:
            title = turkish_item.get("title")
            abstract = turkish_item.get("abstract")
            keywords = normalize_keywords(
                turkish_item.get("keywords")
            )

        article = {
            "article_id": str(
                source.get("id") or hit.get("_id")
            ),
            "publication_language": source.get("language"),
            "publication_year": source.get(
                "publicationYear"
            ),
            "document_type": source.get("docType"),
            "databases": source.get("databases") or [],
            "subjects": source.get("subjects") or [],
            "title_tr": title,
            "abstract_tr": abstract,
            "keywords_tr": keywords,
        }

        extracted_articles.append(article)

    return extracted_articles


def save_extracted_articles(
    articles: List[Dict[str, Any]],
) -> Path:
    """Sadeleştirilmiş kayıtları JSON dosyasına kaydeder."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day05_extracted_articles.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            articles,
            json_file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


def print_article_summary(
    articles: List[Dict[str, Any]],
) -> None:
    """Çıkarılan makaleleri kısa biçimde terminalde gösterir."""

    print("=" * 75)
    print("TR DİZİN TÜRKÇE MAKALE KAYITLARI")
    print("=" * 75)

    print(f"\nToplam çıkarılan kayıt: {len(articles)}")

    abstract_count = sum(
        1
        for article in articles
        if article["abstract_tr"]
    )

    subject_count = sum(
        1
        for article in articles
        if article["subjects"]
    )

    print(f"Türkçe abstract bulunan: {abstract_count}")
    print(f"Subject bulunan kayıt  : {subject_count}")

    for index, article in enumerate(
        articles,
        start=1,
    ):
        abstract = article["abstract_tr"] or ""
        title = article["title_tr"] or "BAŞLIK YOK"

        print("\n" + "-" * 75)
        print(f"Kayıt {index}")
        print(f"ID       : {article['article_id']}")
        print(
            f"Yıl      : {article['publication_year']}"
        )
        print(
            f"Dil      : "
            f"{article['publication_language']}"
        )
        print(
            f"Database : {article['databases']}"
        )
        print(
            f"Subjects : {article['subjects']}"
        )
        print(f"Başlık   : {title}")
        print(
            f"Keywords : {article['keywords_tr']}"
        )
        print(
            f"Özet uzunluğu: {len(abstract)} karakter"
        )

        if abstract:
            print(f"Özet başlangıcı: {abstract[:180]}...")
        else:
            print("Özet başlangıcı: TÜRKÇE ÖZET YOK")


def main() -> None:
    raw_data = load_raw_response()

    articles = extract_articles(raw_data)

    print_article_summary(articles)

    output_path = save_extracted_articles(articles)

    print("\n" + "=" * 75)
    print("ÇIKARMA TAMAMLANDI")
    print("=" * 75)
    print(f"\nSadeleştirilmiş JSON:\n{output_path}")


if __name__ == "__main__":
    main()