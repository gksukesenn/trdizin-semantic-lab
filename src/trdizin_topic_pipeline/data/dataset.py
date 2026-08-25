"""TR Dizin kayıt normalizasyonu, deduplication ve veri seti kontrolleri."""

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from ..utils.io import abstract_sha256


# Noktalama işaretleri çıkarıldıktan ve metin casefold uygulandıktan
# sonra anlamsız placeholder olarak kabul edilecek değerler.
_PLACEHOLDER_ABSTRACT_KEYS = {
    "",
    "boş",
    "bos",
    "özyok",
    "ozyok",
    "özetyok",
    "ozetyok",
    "tözyok",
    "tozyok",
    "türkçeözyok",
    "turkceozyok",
    "türkçeözetyok",
    "turkceozetyok",
    "özetbulunmamaktadır",
    "ozetbulunmamaktadir",
}


def normalize_text(value: Any) -> str:
    """Bir metindeki gereksiz boşlukları temizler."""

    return (
        " ".join(value.split()).strip()
        if isinstance(value, str)
        else ""
    )


def abstract_quality_key(value: Any) -> str:
    """
    Placeholder karşılaştırması için metni sadeleştirir.

    Noktalama ve boşluklar çıkarılır. Türkçe karakterler korunur.
    Örnek:
        "T.Öz Yok" -> "tözyok"
        "--"        -> ""
    """

    text = normalize_text(value).casefold()

    return "".join(
        character
        for character in text
        if character.isalnum()
    )


def is_meaningful_abstract(
    value: Any,
    minimum_characters: int = 1,
) -> bool:
    """
    Metnin gerçek bir abstract adayı olup olmadığını kontrol eder.

    Kısa fakat anlam taşıyan metinler otomatik olarak reddedilmez.
    Yalnızca boş, noktalama-only veya bilinen placeholder değerler elenir.
    """

    text = normalize_text(value)

    if len(text) < minimum_characters:
        return False

    # "-", "--", "...", "_" gibi yalnızca noktalama içeren değerler.
    if not any(character.isalnum() for character in text):
        return False

    quality_key = abstract_quality_key(text)

    if quality_key in _PLACEHOLDER_ABSTRACT_KEYS:
        return False

    return True


def normalize_list(value: Any) -> List[str]:
    """Tek değer veya liste biçimindeki metadata alanını temizler."""

    values = value if isinstance(value, list) else [value]

    return [
        text
        for text in (
            normalize_text(item)
            for item in values
        )
        if text
    ]


def normalize_subjects(value: Any) -> List[Dict[str, Any]]:
    """Subject kayıtlarını sözlük listesine dönüştürür."""

    result: List[Dict[str, Any]] = []

    if not isinstance(value, list):
        return result

    for item in value:
        parsed = item

        if isinstance(item, str):
            try:
                parsed = json.loads(item)
            except json.JSONDecodeError:
                continue

        if isinstance(parsed, dict):
            result.append(parsed)

    return result


def extract_article(
    hit: Any,
    minimum_abstract_characters: int = 1,
) -> Optional[Dict[str, Any]]:
    """TR Dizin arama sonucundan geçerli Türkçe makale kaydı çıkarır."""

    if (
        not isinstance(hit, dict)
        or not isinstance(hit.get("_source"), dict)
    ):
        return None

    source = hit["_source"]
    abstracts = source.get("abstracts", [])

    if not isinstance(abstracts, list):
        return None

    turkish = next(
        (
            item
            for item in abstracts
            if (
                isinstance(item, dict)
                and str(
                    item.get("language", "")
                ).upper() == "TUR"
                and is_meaningful_abstract(
                    item.get("abstract"),
                    minimum_characters=minimum_abstract_characters,
                )
            )
        ),
        None,
    )

    if turkish is None:
        return None

    article_id = str(
        source.get("id")
        or hit.get("_id")
        or ""
    ).strip()

    abstract = normalize_text(
        turkish.get("abstract")
    )

    if (
        not article_id
        or not is_meaningful_abstract(
            abstract,
            minimum_characters=minimum_abstract_characters,
        )
    ):
        return None

    language = str(
        source.get("language")
        or ""
    ).upper()

    document_type = str(
        source.get("docType")
        or ""
    ).upper()

    if language and language != "TUR":
        return None

    if (
        document_type
        and document_type != "PAPER"
    ):
        return None

    return {
        "article_id": article_id,
        "publication_year": source.get(
            "publicationYear"
        ),
        "title_tr": normalize_text(
            turkish.get("title")
        ),
        "abstract_tr": abstract,
        "keywords_tr": normalize_list(
            turkish.get("keywords")
        ),
        "databases": normalize_list(
            source.get("databases")
        ),
        "subjects": normalize_subjects(
            source.get("subjects")
        ),
    }


def exclusion_ids(
    rows: List[Dict[str, Any]],
) -> Set[str]:
    """Veri setindeki geçerli article ID değerlerini döndürür."""

    return {
        str(
            row.get("article_id", "")
        ).strip()
        for row in rows
        if str(
            row.get("article_id", "")
        ).strip()
    }


def dataset_identity(
    rows: List[Dict[str, Any]],
) -> Tuple[Set[str], Set[str]]:
    """Dataset için ID ve anlamlı abstract hash kümelerini oluşturur."""

    ids: Set[str] = set()
    hashes: Set[str] = set()

    for row in rows:
        article_id = str(
            row.get("article_id", "")
        ).strip()

        abstract = normalize_text(
            row.get("abstract_tr")
        )

        if article_id:
            ids.add(article_id)

        if is_meaningful_abstract(abstract):
            hashes.add(
                abstract_sha256(abstract)
            )

    return ids, hashes


def validate_core(
    rows: List[Dict[str, Any]],
    expected: int,
    pilot_ids: Set[str],
    validation_ids: Set[str],
) -> Dict[str, int]:
    """Nihai veri setinin temel bütünlük kurallarını doğrular."""

    ids = [
        str(
            row.get("article_id", "")
        ).strip()
        for row in rows
    ]

    abstracts = [
        normalize_text(
            row.get("abstract_tr")
        )
        for row in rows
    ]

    hashes = [
        abstract_sha256(value)
        for value in abstracts
        if value
    ]

    invalid_abstract_count = sum(
        not is_meaningful_abstract(value)
        for value in abstracts
    )

    issues = {
        "row_count": len(rows),
        "unique_article_id_count": len(
            set(ids)
        ),
        "unique_abstract_sha256_count": len(
            set(hashes)
        ),
        "pilot_id_overlap_count": len(
            set(ids) & pilot_ids
        ),
        "validation_id_overlap_count": len(
            set(ids) & validation_ids
        ),
        "empty_abstract_count": sum(
            not value
            for value in abstracts
        ),
        "invalid_abstract_count": (
            invalid_abstract_count
        ),
        "empty_title_count": sum(
            not normalize_text(
                row.get("title_tr")
            )
            for row in rows
        ),
    }

    failures: List[str] = []

    if len(rows) != expected:
        failures.append(
            "satır=%d (beklenen %d)"
            % (
                len(rows),
                expected,
            )
        )

    if (
        len(set(ids)) != expected
        or any(not value for value in ids)
    ):
        failures.append(
            "article_id benzersizliği/geçerliliği"
        )

    if len(set(hashes)) != expected:
        failures.append(
            "abstract SHA-256 benzersizliği"
        )

    if issues["pilot_id_overlap_count"]:
        failures.append(
            "pilot ID çakışması"
        )

    if issues["validation_id_overlap_count"]:
        failures.append(
            "validation ID çakışması"
        )

    if issues["empty_abstract_count"]:
        failures.append(
            "boş abstract"
        )

    if issues["invalid_abstract_count"]:
        failures.append(
            "placeholder veya anlamsız abstract=%d"
            % issues["invalid_abstract_count"]
        )

    if failures:
        raise ValueError(
            "Veri seti doğrulaması başarısız: %s"
            % "; ".join(failures)
        )

    return issues
