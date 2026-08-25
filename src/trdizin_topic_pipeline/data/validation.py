"""Veri kalitesi istatistikleri ve raporları."""

from collections import Counter
from typing import Any, Dict, List

import numpy as np

from .dataset import is_meaningful_abstract, normalize_text


def numeric_summary(values: List[int]) -> Dict[str, float]:
    """Sayısal değerler için temel dağılım istatistiklerini hesaplar."""

    if not values:
        raise ValueError("Sayısal özet için boş liste verildi.")

    array = np.asarray(values, dtype=np.float64)

    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def database_category(row: Dict[str, Any]) -> str:
    """Makaleyi SCIENCE, SOCIAL, BOTH veya OTHER grubuna ayırır."""

    values = {
        str(value).upper()
        for value in row.get("databases", [])
    }

    if "SCIENCE" in values and "SOCIAL" in values:
        return "BOTH"

    if "SCIENCE" in values:
        return "SCIENCE"

    if "SOCIAL" in values:
        return "SOCIAL"

    return "OTHER"


def metadata_counts(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Datasetin metadata ve abstract kalite istatistiklerini üretir."""

    if not rows:
        raise ValueError("Metadata raporu için veri seti boş.")

    years = Counter(
        str(row.get("publication_year") or "Bilinmiyor")
        for row in rows
    )

    databases = Counter(
        value
        for row in rows
        for value in row.get("databases", [])
    )

    categories = Counter(
        database_category(row)
        for row in rows
    )

    subject_present_count = sum(
        bool(row.get("subjects"))
        for row in rows
    )

    keyword_present_count = sum(
        bool(row.get("keywords_tr"))
        for row in rows
    )

    invalid_abstract_count = sum(
        not is_meaningful_abstract(
            row.get("abstract_tr")
        )
        for row in rows
    )

    row_count = len(rows)

    return {
        "publication_year_distribution": dict(
            sorted(years.items())
        ),
        "database_distribution": dict(
            sorted(databases.items())
        ),
        "science_social_distribution": dict(
            sorted(categories.items())
        ),
        "subject_present_count": subject_present_count,
        "subject_present_rate": (
            subject_present_count / float(row_count)
        ),
        "keyword_present_count": keyword_present_count,
        "keyword_present_rate": (
            keyword_present_count / float(row_count)
        ),
        "invalid_abstract_count": invalid_abstract_count,
        "invalid_abstract_rate": (
            invalid_abstract_count / float(row_count)
        ),
        "abstract_character_length": numeric_summary(
            [
                len(
                    normalize_text(
                        row.get("abstract_tr")
                    )
                )
                for row in rows
            ]
        ),
    }


def load_tokenizer(model_id: str) -> Any:
    """Embedding modelinin tokenizerını yükler."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=False,
    )


def token_lengths(
    rows: List[Dict[str, Any]],
    model_id: str,
) -> List[int]:
    """
    Abstractların kesilmeden önceki gerçek token uzunluklarını hesaplar.

    Tokenizerın 512 üzeri metinler için gereksiz uyarı vermesini önler.
    Embedding aşamasındaki gerçek 512 token sınırını değiştirmez.
    """

    tokenizer = load_tokenizer(model_id)

    original_model_max_length = tokenizer.model_max_length
    tokenizer.model_max_length = 1_000_000

    try:
        return [
            len(
                tokenizer.encode(
                    str(row["abstract_tr"]),
                    add_special_tokens=True,
                    truncation=False,
                )
            )
            for row in rows
        ]
    finally:
        tokenizer.model_max_length = (
            original_model_max_length
        )


def subject_names(
    row: Dict[str, Any],
) -> List[str]:
    """Bir makalenin okunabilir subject isimlerini döndürür."""

    result: List[str] = []

    for item in row.get("subjects", []):
        if not isinstance(item, dict):
            continue

        value = str(
            item.get("fullName")
            or item.get("name")
            or ""
        ).strip()

        if value:
            result.append(value)

    return result