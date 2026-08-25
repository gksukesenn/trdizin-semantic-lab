import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import requests


SEARCH_URL = (
    "https://search.trdizin.gov.tr/"
    "api/defaultSearch/publication/"
)

QUERY = "a"
PAGE_LIMIT = 100
PAGES_PER_ORDER = 3

ORDERS = [
    "publicationYear-DESC",
    "publicationYear-ASC",
]


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


def normalize_subjects(value: Any) -> List[str]:
    """
    Subject değerini her durumda string listesine dönüştürür.

    Alan bazen:
    - null
    - liste
    - string
    - sözlük

    olabilir. Şimdilik gözlem amacıyla güvenli biçimde normalize ediyoruz.
    """

    if value is None:
        return []

    if isinstance(value, str):
        cleaned_value = value.strip()
        return [cleaned_value] if cleaned_value else []

    if isinstance(value, list):
        subjects: List[str] = []

        for item in value:
            if isinstance(item, str):
                cleaned_item = item.strip()

                if cleaned_item:
                    subjects.append(cleaned_item)

            elif isinstance(item, dict):
                subjects.append(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )

        return subjects

    if isinstance(value, dict):
        return [
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
            )
        ]

    return [str(value)]


def fetch_page(
    order: str,
    page: int,
) -> Dict[str, Any]:
    """TR Dizin'den bir arama sayfası getirir."""

    params = {
        "q": QUERY,
        "order": order,
        "page": page,
        "limit": PAGE_LIMIT,
        "facet-documentType": "PAPER",
        "facet-publicationLanguage": "TUR",
    }

    print(
        f"İstek: order={order}, "
        f"page={page}, limit={PAGE_LIMIT}"
    )

    response = requests.get(
        SEARCH_URL,
        params=params,
        timeout=60,
        headers={
            "Accept": "application/json",
            "User-Agent": "trdizin-semantic-lab/0.1",
        },
    )

    print(f"HTTP durum kodu: {response.status_code}")

    response.raise_for_status()

    return response.json()


def collect_records() -> List[Dict[str, Any]]:
    """
    En yeni ve en eski kayıtları toplar.

    Aynı makalenin tekrar gelmesi ihtimaline karşı
    article_id üzerinden tekrarları kaldırır.
    """

    records_by_id: Dict[str, Dict[str, Any]] = {}

    for order in ORDERS:
        for page in range(1, PAGES_PER_ORDER + 1):
            data = fetch_page(
                order=order,
                page=page,
            )

            hits = (
                data
                .get("hits", {})
                .get("hits", [])
            )

            for hit in hits:
                if not isinstance(hit, dict):
                    continue

                source = hit.get("_source", {})

                if not isinstance(source, dict):
                    continue

                article_id = str(
                    source.get("id")
                    or hit.get("_id")
                    or ""
                )

                if not article_id:
                    continue

                subjects = normalize_subjects(
                    source.get("subjects")
                )

                record = {
                    "article_id": article_id,
                    "order_group": order,
                    "publication_year": source.get(
                        "publicationYear"
                    ),
                    "publication_language": source.get(
                        "language"
                    ),
                    "databases": source.get(
                        "databases"
                    ) or [],
                    "subjects": subjects,
                    "has_subject": bool(subjects),
                }

                records_by_id[article_id] = record

    return list(records_by_id.values())


def print_report(
    records: List[Dict[str, Any]],
) -> None:
    """Subject doluluk raporunu terminalde gösterir."""

    total_count = len(records)

    subject_records = [
        record
        for record in records
        if record["has_subject"]
    ]

    subject_count = len(subject_records)

    coverage_percentage = (
        subject_count / total_count * 100
        if total_count
        else 0
    )

    print("\n" + "=" * 75)
    print("SUBJECT KAPSAM RAPORU")
    print("=" * 75)

    print(f"\nToplam benzersiz makale : {total_count}")
    print(f"Subject bulunan makale : {subject_count}")
    print(
        f"Subject doluluk oranı  : "
        f"%{coverage_percentage:.2f}"
    )

    year_total_counter = Counter(
        record["publication_year"]
        for record in records
    )

    year_subject_counter = Counter(
        record["publication_year"]
        for record in subject_records
    )

    print("\nYıllara göre dağılım:")

    for year in sorted(
        year_total_counter,
        key=lambda value: (
            value is None,
            value,
        ),
    ):
        total_for_year = year_total_counter[year]
        subject_for_year = year_subject_counter[year]

        percentage = (
            subject_for_year / total_for_year * 100
            if total_for_year
            else 0
        )

        print(
            f"- {year}: "
            f"{subject_for_year}/{total_for_year} "
            f"(%{percentage:.2f})"
        )

    database_total_counter: Counter[str] = Counter()
    database_subject_counter: Counter[str] = Counter()

    for record in records:
        for database in record["databases"]:
            database_total_counter[database] += 1

            if record["has_subject"]:
                database_subject_counter[database] += 1

    print("\nDatabase türüne göre dağılım:")

    for database, total in database_total_counter.items():
        subject_total = database_subject_counter[database]

        percentage = (
            subject_total / total * 100
            if total
            else 0
        )

        print(
            f"- {database}: "
            f"{subject_total}/{total} "
            f"(%{percentage:.2f})"
        )

    print("\nSubject bulunan ilk 10 örnek:")

    if not subject_records:
        print("- Subject bulunan kayıt yok.")

    for record in subject_records[:10]:
        print(
            f"- ID={record['article_id']} "
            f"| yıl={record['publication_year']} "
            f"| subjects={record['subjects']}"
        )


def save_outputs(
    records: List[Dict[str, Any]],
) -> None:
    """Sonuçları JSON ve CSV olarak kaydeder."""

    output_directory = (
        get_project_root()
        / "research" / "outputs"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_directory
        / "day07_subject_coverage.json"
    )

    csv_path = (
        output_directory
        / "day07_subject_coverage.csv"
    )

    with json_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            records,
            json_file,
            ensure_ascii=False,
            indent=2,
        )

    with csv_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "article_id",
                "order_group",
                "publication_year",
                "publication_language",
                "databases",
                "subjects",
                "has_subject",
            ],
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    **record,
                    "databases": json.dumps(
                        record["databases"],
                        ensure_ascii=False,
                    ),
                    "subjects": json.dumps(
                        record["subjects"],
                        ensure_ascii=False,
                    ),
                }
            )

    print(f"\nJSON raporu:\n{json_path}")
    print(f"\nCSV raporu:\n{csv_path}")


def main() -> None:
    records = collect_records()

    print_report(records)
    save_outputs(records)


if __name__ == "__main__":
    main()