import csv
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------------------------------------------------
# 1. TR Dizin API ayarları
# ---------------------------------------------------------

SEARCH_URL = (
    "https://search.trdizin.gov.tr/"
    "api/defaultSearch/publication/"
)

# Her yıldan yaklaşık 100 geçerli Türkçe abstract alacağız.
YEARS = [
    2010,
    2012,
    2014,
    2016,
    2018,
    2020,
    2021,
    2022,
    2024,
    2026,
]

TARGET_PER_YEAR = 100

# Tek sayfada TR Dizin'in izin verdiği en yüksek kayıt sayısı.
PAGE_LIMIT = 100

# Her sorgu için en fazla iki sayfa inceleyeceğiz.
PAGES_PER_QUERY = 2

# "a" yeterli sonuç üretmezse "e" sorgusunu da deneyeceğiz.
#
# Bu sorgular nihai veri seti toplama yöntemi değildir.
# Yalnızca model benchmark'ı için çeşitli bir pilot veri
# elde etmek amacıyla kullanılmaktadır.
QUERY_TERMS = [
    "a",
    "e",
]

# Çok kısa metinleri akademik abstract olarak kabul etmiyoruz.
MIN_ABSTRACT_CHARACTERS = 200

# API'ye arka arkaya çok hızlı istek göndermemek için bekleme.
REQUEST_DELAY_SECONDS = 0.4

RANDOM_SEED = 42


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


def create_http_session() -> requests.Session:
    """
    Geçici sunucu hatalarında isteği yeniden deneyen
    bir HTTP oturumu oluşturur.
    """

    retry_strategy = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=frozenset(["GET"]),
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter,
    )

    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "trdizin-semantic-lab/0.1",
        }
    )

    return session


def normalize_text(value: Any) -> str:
    """
    Metindeki gereksiz satır sonlarını ve art arda
    gelen boşlukları temizler.
    """

    if not isinstance(value, str):
        return ""

    return " ".join(value.split()).strip()


def normalize_keywords(value: Any) -> List[str]:
    """Keywords alanını temiz bir string listesine dönüştürür."""

    if isinstance(value, list):
        keywords: List[str] = []

        for item in value:
            cleaned_item = normalize_text(item)

            if cleaned_item:
                keywords.append(cleaned_item)

        return keywords

    if isinstance(value, str):
        cleaned_value = normalize_text(value)

        if cleaned_value:
            return [cleaned_value]

    return []


def normalize_subjects(value: Any) -> List[Dict[str, Any]]:
    """
    Subject alanını sözlüklerden oluşan liste biçimine getirir.

    Subjectleri şu anda analiz etmiyoruz.
    Yalnızca ileride kullanmak üzere saklıyoruz.
    """

    if not isinstance(value, list):
        return []

    subjects: List[Dict[str, Any]] = []

    for item in value:
        if isinstance(item, dict):
            subjects.append(item)
            continue

        # Önceki bazı dosyalarda subject sözlükleri
        # JSON stringi biçiminde bulunabilir.
        if isinstance(item, str):
            try:
                parsed_item = json.loads(item)
            except json.JSONDecodeError:
                continue

            if isinstance(parsed_item, dict):
                subjects.append(parsed_item)

    return subjects


def find_turkish_abstract(
    abstract_items: Any,
) -> Optional[Dict[str, Any]]:
    """
    Abstract kayıtları arasından TUR dilindeki
    başlık, özet ve anahtar kelime kaydını bulur.
    """

    if not isinstance(abstract_items, list):
        return None

    for abstract_item in abstract_items:
        if not isinstance(abstract_item, dict):
            continue

        if abstract_item.get("language") == "TUR":
            return abstract_item

    return None


def fetch_page(
    session: requests.Session,
    year: int,
    query: str,
    page: int,
) -> Dict[str, Any]:
    """Belirli yıl, sorgu ve sayfa için API isteği gönderir."""

    params = {
        "q": query,
        "order": "publicationYear-DESC",
        "page": page,
        "limit": PAGE_LIMIT,
        "facet-documentType": "PAPER",
        "facet-publicationLanguage": "TUR",
        "facet-publication_year": year,
    }

    print(
        f"İstek gönderiliyor: "
        f"yıl={year}, sorgu={query!r}, sayfa={page}"
    )

    response = session.get(
        SEARCH_URL,
        params=params,
        timeout=60,
    )

    print(
        f"HTTP durum kodu: "
        f"{response.status_code}"
    )

    response.raise_for_status()

    time.sleep(REQUEST_DELAY_SECONDS)

    return response.json()


def extract_article(
    hit: Any,
) -> Optional[Dict[str, Any]]:
    """
    Ham TR Dizin hit kaydından standart makale
    kaydımızı oluşturur.
    """

    if not isinstance(hit, dict):
        return None

    source = hit.get("_source")

    if not isinstance(source, dict):
        return None

    turkish_abstract_item = find_turkish_abstract(
        source.get("abstracts")
    )

    if turkish_abstract_item is None:
        return None

    title_tr = normalize_text(
        turkish_abstract_item.get("title")
    )

    abstract_tr = normalize_text(
        turkish_abstract_item.get("abstract")
    )

    if len(abstract_tr) < MIN_ABSTRACT_CHARACTERS:
        return None

    article_id = str(
        source.get("id")
        or hit.get("_id")
        or ""
    ).strip()

    if not article_id:
        return None

    return {
        "article_id": article_id,
        "title_tr": title_tr,
        "abstract_tr": abstract_tr,
        "keywords_tr": normalize_keywords(
            turkish_abstract_item.get("keywords")
        ),
        "publication_year": source.get(
            "publicationYear"
        ),
        "publication_language": source.get(
            "language"
        ),
        "document_type": source.get(
            "docType"
        ),
        "databases": source.get(
            "databases"
        ) or [],
        "subjects": normalize_subjects(
            source.get("subjects")
        ),
        "abstract_character_count": len(
            abstract_tr
        ),
    }


def collect_candidates_for_year(
    session: requests.Session,
    year: int,
) -> List[Dict[str, Any]]:
    """
    Bir yıl için farklı sorgu ve sayfalardan
    geçerli makale adayları toplar.
    """

    candidates_by_id: Dict[str, Dict[str, Any]] = {}

    for query in QUERY_TERMS:
        for page in range(
            1,
            PAGES_PER_QUERY + 1,
        ):
            data = fetch_page(
                session=session,
                year=year,
                query=query,
                page=page,
            )

            hits = (
                data
                .get("hits", {})
                .get("hits", [])
            )

            if not isinstance(hits, list):
                continue

            for hit in hits:
                article = extract_article(hit)

                if article is None:
                    continue

                candidates_by_id[
                    article["article_id"]
                ] = article

            # Hedefin iki katına yaklaştıysak
            # yeterli örnek havuzumuz vardır.
            if len(candidates_by_id) >= (
                TARGET_PER_YEAR * 2
            ):
                break

        if len(candidates_by_id) >= (
            TARGET_PER_YEAR * 2
        ):
            break

    return list(
        candidates_by_id.values()
    )


def select_articles_for_year(
    candidates: List[Dict[str, Any]],
    year: int,
) -> List[Dict[str, Any]]:
    """
    Adaylar arasından sabit random seed kullanarak
    en fazla 100 makale seçer.

    Sabit seed, aynı veriden tekrar aynı örneğin
    üretilebilmesini sağlar.
    """

    candidates = sorted(
        candidates,
        key=lambda article: article["article_id"],
    )

    if len(candidates) <= TARGET_PER_YEAR:
        return candidates

    random_generator = random.Random(
        RANDOM_SEED + year
    )

    selected_articles = random_generator.sample(
        candidates,
        TARGET_PER_YEAR,
    )

    return sorted(
        selected_articles,
        key=lambda article: article["article_id"],
    )


def build_pilot_dataset() -> List[Dict[str, Any]]:
    """Bütün yıllardan pilot makaleleri toplar."""

    session = create_http_session()

    articles_by_id: Dict[str, Dict[str, Any]] = {}

    print("=" * 75)
    print("TR DİZİN TÜRKÇE PİLOT VERİ SETİ")
    print("=" * 75)

    try:
        for year in YEARS:
            print("\n" + "=" * 75)
            print(f"YIL: {year}")
            print("=" * 75)

            candidates = collect_candidates_for_year(
                session=session,
                year=year,
            )

            selected_articles = select_articles_for_year(
                candidates=candidates,
                year=year,
            )

            print(
                f"\nGeçerli aday sayısı : "
                f"{len(candidates)}"
            )

            print(
                f"Seçilen makale sayısı: "
                f"{len(selected_articles)}"
            )

            for article in selected_articles:
                articles_by_id[
                    article["article_id"]
                ] = article

    finally:
        session.close()

    return sorted(
        articles_by_id.values(),
        key=lambda article: (
            article["publication_year"] or 0,
            article["article_id"],
        ),
    )


def save_jsonl(
    articles: List[Dict[str, Any]],
) -> Path:
    """
    Makaleleri JSONL biçiminde kaydeder.

    JSONL dosyasında her satır bağımsız bir
    JSON makale kaydıdır.
    """

    output_path = (
        get_project_root()
        / "data"
        / "processed"
        / "pilot_articles_1000.jsonl"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as output_file:
        for article in articles:
            json.dump(
                article,
                output_file,
                ensure_ascii=False,
            )

            output_file.write("\n")

    return output_path


def save_preview_csv(
    articles: List[Dict[str, Any]],
) -> Path:
    """İlk 50 makaleyi kolay incelemek için CSV'ye yazar."""

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day08_pilot_preview.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        fieldnames = [
            "article_id",
            "publication_year",
            "databases",
            "title_tr",
            "keywords_tr",
            "abstract_character_count",
            "abstract_preview",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for article in articles[:50]:
            writer.writerow(
                {
                    "article_id": article[
                        "article_id"
                    ],
                    "publication_year": article[
                        "publication_year"
                    ],
                    "databases": json.dumps(
                        article["databases"],
                        ensure_ascii=False,
                    ),
                    "title_tr": article["title_tr"],
                    "keywords_tr": json.dumps(
                        article["keywords_tr"],
                        ensure_ascii=False,
                    ),
                    "abstract_character_count": article[
                        "abstract_character_count"
                    ],
                    "abstract_preview": article[
                        "abstract_tr"
                    ][:250],
                }
            )

    return output_path


def save_summary(
    articles: List[Dict[str, Any]],
) -> Path:
    """Pilot veri setinin temel teknik özetini kaydeder."""

    year_counter = Counter(
        article["publication_year"]
        for article in articles
    )

    database_counter: Counter[str] = Counter()

    for article in articles:
        for database in article["databases"]:
            database_counter[str(database)] += 1

    abstract_lengths = [
        article["abstract_character_count"]
        for article in articles
    ]

    summary = {
        "total_article_count": len(articles),
        "requested_article_count": (
            len(YEARS) * TARGET_PER_YEAR
        ),
        "year_distribution": dict(
            sorted(year_counter.items())
        ),
        "database_distribution": dict(
            database_counter
        ),
        "abstract_character_statistics": {
            "minimum": (
                min(abstract_lengths)
                if abstract_lengths
                else 0
            ),
            "maximum": (
                max(abstract_lengths)
                if abstract_lengths
                else 0
            ),
            "mean": (
                statistics.mean(abstract_lengths)
                if abstract_lengths
                else 0
            ),
            "median": (
                statistics.median(abstract_lengths)
                if abstract_lengths
                else 0
            ),
        },
        "years": YEARS,
        "target_per_year": TARGET_PER_YEAR,
        "minimum_abstract_characters": (
            MIN_ABSTRACT_CHARACTERS
        ),
        "sampling_note": (
            "Bu dosya embedding modeli benchmark'i için "
            "hazırlanan teknik pilot örneklemdir. TR Dizin'in "
            "genel dağılımını temsil ettiği varsayılmamalıdır."
        ),
    }

    output_path = (
        get_project_root()
        / "research" / "outputs"
        / "day08_pilot_summary.json"
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
            summary,
            json_file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


def main() -> None:
    articles = build_pilot_dataset()

    jsonl_path = save_jsonl(articles)
    preview_path = save_preview_csv(articles)
    summary_path = save_summary(articles)

    print("\n" + "=" * 75)
    print("PİLOT VERİ SETİ TAMAMLANDI")
    print("=" * 75)

    print(
        f"\nToplam geçerli makale: "
        f"{len(articles)}"
    )

    print(f"\nJSONL veri dosyası:\n{jsonl_path}")
    print(f"\nCSV ön izleme:\n{preview_path}")
    print(f"\nTeknik özet:\n{summary_path}")


if __name__ == "__main__":
    main()

'''
TR DİZİN TÜRKÇE PİLOT VERİ SETİ
===========================================================================

===========================================================================
YIL: 2010
===========================================================================
İstek gönderiliyor: yıl=2010, sorgu='a', sayfa=1
HTTP durum kodu: 200
İstek gönderiliyor: yıl=2010, sorgu='a', sayfa=2
HTTP durum kodu: 200

Geçerli aday sayısı : 200
Seçilen makale sayısı: 100

===========================================================================
YIL: 2012
===========================================================================
İstek gönderiliyor: yıl=2012, sorgu='a', sayfa=1
HTTP durum kodu: 200
İstek gönderiliyor: yıl=2012, sorgu='a', sayfa=2
HTTP durum kodu: 200
İstek gönderiliyor: yıl=2012, sorgu='e', sayfa=1
HTTP durum kodu: 200

Geçerli aday sayısı : 283
Seçilen makale sayısı: 100

===========================================================================
YIL: 2014
===========================================================================
İstek gönderiliyor: yıl=2014, sorgu='a', sayfa=1
HTTP durum kodu: 200
İstek gönderiliyor: yıl=2014, sorgu='a', sayfa=2
HTTP durum kodu: 200
İstek gönderiliyor: yıl=2014, sorgu='e', sayfa=1
HTTP durum kodu: 200

Geçerli aday sayısı : 289
Seçilen makale sayısı: 100

===========================================================================
YIL: 2016
===========================================================================
İstek gönderiliyor: yıl=2016, sorgu='a', sayfa=1
HTTP durum kodu: 200
İstek gönderiliyor: yıl=2016, sorgu='a', sayfa=2
HTTP durum kodu: 200

Geçerli aday sayısı : 200
Seçilen makale sayısı: 100

===========================================================================
YIL: 2018
===========================================================================
İstek gönderiliyor: yıl=2018, sorgu='a', sayfa=1
HTTP durum kodu: 200
İstek gönderiliyor: yıl=2018, sorgu='a', sayfa=2
HTTP durum kodu: 200
İstek gönderiliyor: yıl=2018, sorgu='e', sayfa=1
HTTP durum kodu: 200

Geçerli aday sayısı : 289
Seçilen makale sayısı: 100

===========================================================================
YIL: 2020
===========================================================================
İstek gönderiliyor: yıl=2020, sorgu='a', sayfa=1
HTTP durum kodu: 200
İstek gönderiliyor: yıl=2020, sorgu='a', sayfa=2
HTTP durum kodu: 200
İstek gönderiliyor: yıl=2020, sorgu='e', sayfa=1
HTTP durum kodu: 200

Geçerli aday sayısı : 291
Seçilen makale sayısı: 100

===========================================================================
YIL: 2021
===========================================================================
İstek gönderiliyor: yıl=2021, sorgu='a', sayfa=1
HTTP durum kodu: 200
İstek gönderiliyor: yıl=2021, sorgu='a', sayfa=2
HTTP durum kodu: 200
İstek gönderiliyor: yıl=2021, sorgu='e', sayfa=1
HTTP durum kodu: 200

Geçerli aday sayısı : 293
Seçilen makale sayısı: 100

===========================================================================
YIL: 2022
===========================================================================
İstek gönderiliyor: yıl=2022, sorgu='a', sayfa=1
HTTP durum kodu: 200
İstek gönderiliyor: yıl=2022, sorgu='a', sayfa=2
HTTP durum kodu: 200

Geçerli aday sayısı : 200
Seçilen makale sayısı: 100

===========================================================================
YIL: 2024
===========================================================================
İstek gönderiliyor: yıl=2024, sorgu='a', sayfa=1
HTTP durum kodu: 200
İstek gönderiliyor: yıl=2024, sorgu='a', sayfa=2
HTTP durum kodu: 200

Geçerli aday sayısı : 200
Seçilen makale sayısı: 100

===========================================================================
YIL: 2026
===========================================================================
İstek gönderiliyor: yıl=2026, sorgu='a', sayfa=1
HTTP durum kodu: 200
İstek gönderiliyor: yıl=2026, sorgu='a', sayfa=2
HTTP durum kodu: 200

Geçerli aday sayısı : 200
Seçilen makale sayısı: 100

===========================================================================
PİLOT VERİ SETİ TAMAMLANDI
===========================================================================

Toplam geçerli makale: 1000

JSONL veri dosyası:
/home/goksu/trdizin-semantic-lab/data/processed/pilot_articles_1000.jsonl

CSV ön izleme:
/home/goksu/trdizin-semantic-lab/research/outputs/day08_pilot_preview.csv

Teknik özet:
/home/goksu/trdizin-semantic-lab/research/outputs/day08_pilot_summary.json
'''