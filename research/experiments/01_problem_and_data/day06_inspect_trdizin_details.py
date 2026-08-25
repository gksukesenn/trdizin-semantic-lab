import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests


DETAIL_URL_TEMPLATE = (
    "https://search.trdizin.gov.tr/"
    "api/publicationById/{article_id}"
)

ARTICLE_IDS = [
    "1451288",  # SCIENCE: gıda/protein kimyası
    "1450480",  # SOCIAL: matematik eğitimi
    "1450478",  # SOCIAL: yapay zekâ tutum ölçeği
]

INTERESTING_KEYWORDS = (
    "subject",
    "category",
    "classification",
    "discipline",
    "field",
    "topic",
    "sciencefield",
)


def get_project_root() -> Path:
    """Projenin ana klasörünü döndürür."""

    return Path(__file__).resolve().parents[3]


def fetch_article_detail(article_id: str) -> Any:
    """Bir TR Dizin makalesinin detay cevabını alır."""

    url = DETAIL_URL_TEMPLATE.format(
        article_id=article_id
    )

    print("\n" + "=" * 75)
    print(f"MAKALE DETAY İSTEĞİ: {article_id}")
    print("=" * 75)
    print(f"\nAdres:\n{url}")

    response = requests.get(
        url,
        timeout=60,
        headers={
            "Accept": "application/json",
            "User-Agent": "trdizin-semantic-lab/0.1",
        },
    )

    print(f"HTTP durum kodu: {response.status_code}")

    response.raise_for_status()

    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as error:
        print("\nCevap JSON değil.")
        print(response.text[:500])

        raise RuntimeError(
            f"{article_id} için geçerli JSON alınamadı."
        ) from error


def save_raw_detail(
    article_id: str,
    data: Any,
) -> Path:
    """Ham detay cevabını ayrı JSON dosyasına kaydeder."""

    output_path = (
        get_project_root()
        / "data"
        / "raw"
        / "day06_details"
        / f"{article_id}.json"
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


def find_interesting_fields(
    value: Any,
    current_path: str = "root",
) -> List[Tuple[str, Any]]:
    """
    JSON içinde subject, category, field gibi
    olası sınıflandırma alanlarını arar.
    """

    found_fields: List[Tuple[str, Any]] = []

    if isinstance(value, dict):
        for key, child_value in value.items():
            child_path = f"{current_path}.{key}"

            normalized_key = key.lower()

            if any(
                keyword in normalized_key
                for keyword in INTERESTING_KEYWORDS
            ):
                found_fields.append(
                    (child_path, child_value)
                )

            found_fields.extend(
                find_interesting_fields(
                    child_value,
                    child_path,
                )
            )

    elif isinstance(value, list):
        for index, child_value in enumerate(value):
            child_path = f"{current_path}[{index}]"

            found_fields.extend(
                find_interesting_fields(
                    child_value,
                    child_path,
                )
            )

    return found_fields


def print_top_level_structure(data: Any) -> None:
    """Detay cevabının üst seviyesini gösterir."""

    print(f"\nAna veri tipi: {type(data).__name__}")

    if isinstance(data, dict):
        print("En üst seviye anahtarları:")

        for key, value in data.items():
            print(
                f"- {key}: {type(value).__name__}"
            )


def print_interesting_fields(data: Any) -> None:
    """Bulunan olası konu alanlarını yazdırır."""

    fields = find_interesting_fields(data)

    print("\nOlası konu/sınıflandırma alanları:")

    if not fields:
        print("- Hiçbir aday alan bulunamadı.")
        return

    for path, value in fields:
        print(f"\n- Yol: {path}")
        print(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
            )
        )


def main() -> None:
    successful_count = 0

    for article_id in ARTICLE_IDS:
        try:
            data = fetch_article_detail(article_id)
        except requests.RequestException as error:
            print(f"\nİstek başarısız oldu: {error}")
            continue

        successful_count += 1

        output_path = save_raw_detail(
            article_id,
            data,
        )

        print_top_level_structure(data)
        print_interesting_fields(data)

        print(f"\nHam detay dosyası:\n{output_path}")

    print("\n" + "=" * 75)
    print("DETAY İNCELEMESİ TAMAMLANDI")
    print("=" * 75)
    print(
        f"\nBaşarılı detay isteği: "
        f"{successful_count}/{len(ARTICLE_IDS)}"
    )


if __name__ == "__main__":
    main()