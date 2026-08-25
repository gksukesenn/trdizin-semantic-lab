import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests


# ---------------------------------------------------------
# 1. TR Dizin arama adresi
# ---------------------------------------------------------

SEARCH_URL = (
    "https://search.trdizin.gov.tr/"
    "api/defaultSearch/publication/"
)


# ---------------------------------------------------------
# 2. İlk inceleme isteğinin parametreleri
# ---------------------------------------------------------
#
# Bu aşamada yalnızca 10 kayıt istiyoruz.
#
# q="a":
# Şema incelemek için kullanılan geniş bir örnek sorgudur.
# Bu sorguyla gelen kayıtlar nihai veri setimiz olmayacak.
#
# facet-publicationLanguage="TUR":
# Yayın dili Türkçe olan kayıtları istemeye çalışıyoruz.
#

SEARCH_PARAMS = {
    "q": "a",
    "order": "publicationYear-DESC",
    "page": 1,
    "limit": 10,
    "facet-documentType": "PAPER",
    "facet-publicationLanguage": "TUR",
}


def get_project_root() -> Path:
    """
    Projenin ana klasörünü döndürür.

    Bu dosya:
    proje/research/experiments/01_problem_and_data/day04_inspect_trdizin_search.py

    parents[3]:
    proje/
    """

    return Path(__file__).resolve().parents[3]


def fetch_search_response() -> Any:
    """
    TR Dizin API'ye arama isteği gönderir
    ve dönen JSON verisini Python nesnesine çevirir.
    """

    print("=" * 75)
    print("1. TR DİZİN ARAMA İSTEĞİ")
    print("=" * 75)

    print(f"\nAdres: {SEARCH_URL}")
    print("\nParametreler:")

    for parameter_name, parameter_value in SEARCH_PARAMS.items():
        print(f"- {parameter_name}: {parameter_value}")

    print("\nİstek gönderiliyor...")

    response = requests.get(
        SEARCH_URL,
        params=SEARCH_PARAMS,
        timeout=60,
        headers={
            "Accept": "application/json",
            "User-Agent": "trdizin-semantic-lab/0.1",
        },
    )

    print(f"HTTP durum kodu: {response.status_code}")
    print(f"Gerçek istek adresi:\n{response.url}")

    # 400, 404, 500 gibi bir cevap geldiyse
    # programı açıklayıcı hata ile durdurur.
    response.raise_for_status()

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as error:
        print("\nSunucu geçerli JSON döndürmedi.")
        print("Cevabın ilk 500 karakteri:")
        print(response.text[:500])

        raise RuntimeError(
            "TR Dizin cevabı JSON biçiminde değil."
        ) from error


def save_raw_response(data: Any) -> Path:
    """
    API'den gelen ham JSON cevabını değiştirmeden kaydeder.
    """

    output_path = (
        get_project_root()
        / "data"
        / "raw"
        / "day04_trdizin_search_response.json"
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
            data,
            json_file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


def print_top_level_information(data: Any) -> None:
    """
    JSON cevabının en üst seviyedeki yapısını gösterir.
    """

    print("\n" + "=" * 75)
    print("2. JSON'UN EN ÜST SEVİYESİ")
    print("=" * 75)

    print(f"\nAna veri tipi: {type(data).__name__}")

    if isinstance(data, dict):
        print("\nEn üst seviye anahtarları:")

        for key, value in data.items():
            print(
                f"- {key}: "
                f"{type(value).__name__}"
            )

    elif isinstance(data, list):
        print(f"\nEn üst seviyede {len(data)} elemanlı liste var.")

    else:
        print(f"\nDeğer: {data!r}")


def find_record_lists(
    value: Any,
    current_path: str = "root",
    max_depth: int = 6,
    current_depth: int = 0,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    JSON içinde 'liste içinde sözlükler' biçimindeki alanları arar.

    Neden?
    API cevabındaki makale listesinin tam yolunu henüz
    varsaymak istemiyoruz.

    Örneğin makaleler şu yollardan birinde olabilir:

    root.documents
    root.data.content
    root.result.publications

    Bu fonksiyon olası bütün kayıt listelerini bulmaya çalışır.
    """

    found_lists: List[Tuple[str, List[Dict[str, Any]]]] = []

    if current_depth > max_depth:
        return found_lists

    if isinstance(value, dict):
        for key, child_value in value.items():
            child_path = f"{current_path}.{key}"

            found_lists.extend(
                find_record_lists(
                    value=child_value,
                    current_path=child_path,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                )
            )

    elif isinstance(value, list):
        dictionary_items = [
            item
            for item in value
            if isinstance(item, dict)
        ]

        if dictionary_items:
            found_lists.append(
                (
                    current_path,
                    dictionary_items,
                )
            )

        # Listenin ilk birkaç elemanının altını da inceler.
        for index, child_value in enumerate(value[:3]):
            child_path = f"{current_path}[{index}]"

            found_lists.extend(
                find_record_lists(
                    value=child_value,
                    current_path=child_path,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                )
            )

    return found_lists


def print_candidate_record_lists(data: Any) -> None:
    """
    Makale kayıtlarının bulunabileceği liste alanlarını terminale yazar.
    """

    print("\n" + "=" * 75)
    print("3. OLASI KAYIT LİSTELERİ")
    print("=" * 75)

    candidate_lists = find_record_lists(data)

    if not candidate_lists:
        print(
            "\nJSON içinde sözlüklerden oluşan "
            "bir liste bulunamadı."
        )
        return

    # Aynı yolun tekrar yazılmasını önler.
    displayed_paths = set()

    for path, records in candidate_lists:
        if path in displayed_paths:
            continue

        displayed_paths.add(path)

        first_record = records[0]

        print("\n" + "-" * 75)
        print(f"JSON yolu: {path}")
        print(f"Listedeki sözlük sayısı: {len(records)}")
        print("İlk kaydın anahtarları:")

        for key in first_record.keys():
            print(f"- {key}")


def print_first_candidate_record(data: Any) -> None:
    """
    Bulunan ilk olası kayıt listesindeki
    ilk kaydı okunabilir JSON olarak gösterir.
    """

    print("\n" + "=" * 75)
    print("4. İLK OLASI KAYDIN TAM İÇERİĞİ")
    print("=" * 75)

    candidate_lists = find_record_lists(data)

    if not candidate_lists:
        print("\nGösterilecek aday kayıt bulunamadı.")
        return

    path, records = candidate_lists[0]

    print(f"\nKaydın bulunduğu yol: {path}")
    print("\nİlk kayıt:")

    print(
        json.dumps(
            records[0],
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    data = fetch_search_response()

    raw_file_path = save_raw_response(data)

    print_top_level_information(data)
    print_candidate_record_lists(data)
    print_first_candidate_record(data)

    print("\n" + "=" * 75)
    print("İNCELEME TAMAMLANDI")
    print("=" * 75)

    print(f"\nHam JSON dosyası:\n{raw_file_path}")

    print(
        "\nBu aşamada veriyi dönüştürmedik veya temizlemedik. "
        "Yalnızca API'nin gerçek yapısını inceledik."
    )


if __name__ == "__main__":
    main()

'''
python research/experiments/01_problem_and_data/day04_inspect_trdizin_search.py
===========================================================================
1. TR DİZİN ARAMA İSTEĞİ
===========================================================================

Adres: https://search.trdizin.gov.tr/api/defaultSearch/publication/

Parametreler:
- q: a
- order: publicationYear-DESC
- page: 1
- limit: 10
- facet-documentType: PAPER
- facet-publicationLanguage: TUR

İstek gönderiliyor...
HTTP durum kodu: 200
Gerçek istek adresi:
https://search.trdizin.gov.tr/api/defaultSearch/publication/?q=a&order=publicationYear-DESC&page=1&limit=10&facet-documentType=PAPER&facet-publicationLanguage=TUR

===========================================================================
2. JSON'UN EN ÜST SEVİYESİ
===========================================================================

Ana veri tipi: dict

En üst seviye anahtarları:
- _shards: dict
- aggregations: dict
- hits: dict
- timed_out: bool
- took: int

===========================================================================
3. OLASI KAYIT LİSTELERİ
===========================================================================

---------------------------------------------------------------------------
JSON yolu: root.aggregations.facet-accessType.buckets
Listedeki sözlük sayısı: 2
İlk kaydın anahtarları:
- doc_count
- key

---------------------------------------------------------------------------
JSON yolu: root.aggregations.facet-authorName.values.buckets
Listedeki sözlük sayısı: 1000
İlk kaydın anahtarları:
- doc_count
- key

---------------------------------------------------------------------------
JSON yolu: root.aggregations.facet-database.buckets
Listedeki sözlük sayısı: 2
İlk kaydın anahtarları:
- doc_count
- key

---------------------------------------------------------------------------
JSON yolu: root.aggregations.facet-documentType.buckets
Listedeki sözlük sayısı: 1
İlk kaydın anahtarları:
- doc_count
- key

---------------------------------------------------------------------------
JSON yolu: root.aggregations.facet-facetAuthorCity.buckets
Listedeki sözlük sayısı: 82
İlk kaydın anahtarları:
- doc_count
- key

---------------------------------------------------------------------------
JSON yolu: root.aggregations.facet-facetAuthorCountry.buckets
Listedeki sözlük sayısı: 111
İlk kaydın anahtarları:
- doc_count
- key

---------------------------------------------------------------------------
JSON yolu: root.aggregations.facet-facetAuthorInstitution.buckets
Listedeki sözlük sayısı: 483
İlk kaydın anahtarları:
- doc_count
- key

---------------------------------------------------------------------------
JSON yolu: root.aggregations.facet-journalName.values.buckets
Listedeki sözlük sayısı: 1000
İlk kaydın anahtarları:
- doc_count
- key

---------------------------------------------------------------------------
JSON yolu: root.aggregations.facet-publicationLanguage.buckets
Listedeki sözlük sayısı: 1
İlk kaydın anahtarları:
- doc_count
- key

---------------------------------------------------------------------------
JSON yolu: root.aggregations.facet-publicationType.buckets
Listedeki sözlük sayısı: 15
İlk kaydın anahtarları:
- doc_count
- key

---------------------------------------------------------------------------
JSON yolu: root.aggregations.facet-publication_year.values.buckets
Listedeki sözlük sayısı: 36
İlk kaydın anahtarları:
- doc_count
- key

---------------------------------------------------------------------------
JSON yolu: root.aggregations.facet-subject.values.buckets
Listedeki sözlük sayısı: 197
İlk kaydın anahtarları:
- doc_count
- key

---------------------------------------------------------------------------
JSON yolu: root.hits.hits
Listedeki sözlük sayısı: 10
İlk kaydın anahtarları:
- _id
- _index
- _score
- _source
- highlight
- sort

---------------------------------------------------------------------------
JSON yolu: root.hits.hits[0]._source.abstracts
Listedeki sözlük sayısı: 2
İlk kaydın anahtarları:
- abstract
- id
- keywords
- language
- title

---------------------------------------------------------------------------
JSON yolu: root.hits.hits[0]._source.authors
Listedeki sözlük sayısı: 3
İlk kaydın anahtarları:
- authorId
- duty
- inPublicationName
- institution
- institutionName
- isVerified
- name
- orcid
- order
- relationId
- userId

---------------------------------------------------------------------------
JSON yolu: root.hits.hits[0]._source.references
Listedeki sözlük sayısı: 37
İlk kaydın anahtarları:
- authors
- context
- id
- journalCode
- order
- targetPublication
- year

---------------------------------------------------------------------------
JSON yolu: root.hits.hits[1]._source.abstracts
Listedeki sözlük sayısı: 2
İlk kaydın anahtarları:
- abstract
- id
- keywords
- language
- title

---------------------------------------------------------------------------
JSON yolu: root.hits.hits[1]._source.authors
Listedeki sözlük sayısı: 2
İlk kaydın anahtarları:
- authorId
- duty
- inPublicationName
- institution
- institutionName
- isVerified
- name
- orcid
- order
- relationId
- userId

---------------------------------------------------------------------------
JSON yolu: root.hits.hits[1]._source.references
Listedeki sözlük sayısı: 34
İlk kaydın anahtarları:
- authors
- context
- id
- journalCode
- order
- targetPublication
- year

---------------------------------------------------------------------------
JSON yolu: root.hits.hits[2]._source.abstracts
Listedeki sözlük sayısı: 2
İlk kaydın anahtarları:
- abstract
- id
- keywords
- language
- title

---------------------------------------------------------------------------
JSON yolu: root.hits.hits[2]._source.authors
Listedeki sözlük sayısı: 4
İlk kaydın anahtarları:
- authorId
- duty
- inPublicationName
- institution
- institutionName
- isVerified
- name
- orcid
- order
- relationId
- userId

---------------------------------------------------------------------------
JSON yolu: root.hits.hits[2]._source.references
Listedeki sözlük sayısı: 40
İlk kaydın anahtarları:
- authors
- context
- id
- journalCode
- order
- targetPublication
- year

===========================================================================
4. İLK OLASI KAYDIN TAM İÇERİĞİ
===========================================================================

Kaydın bulunduğu yol: root.aggregations.facet-accessType.buckets

İlk kayıt:
{
  "doc_count": 325586,
  "key": "OPEN"
}

===========================================================================
İNCELEME TAMAMLANDI
===========================================================================

Ham JSON dosyası:
/home/goksu/trdizin-semantic-lab/data/raw/day04_trdizin_search_response.json

Bu aşamada veriyi dönüştürmedik veya temizlemedik. Yalnızca API'nin gerçek yapısını inceledik.

'''
