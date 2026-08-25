"""Stable file, scalar and point-ID helpers shared by all indexers."""

import csv
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from ..utils.io import read_json


def read_articles(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("JSONL satırı geçersiz: %d" % line_number) from error
            if not isinstance(row, dict):
                raise ValueError("JSONL satırı sözlük değil: %d" % line_number)
            rows.append(row)
    return rows


def read_assignments(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def optional_int(value: Any) -> Optional[int]:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def optional_float(value: Any) -> Optional[float]:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def point_id(article_id: str) -> Any:
    stripped = article_id.strip()
    if stripped.isdigit():
        integer_id = int(stripped)
        if 0 <= integer_id < 2**64:
            return integer_id
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "trdizin:" + stripped))


def subject_names(article: Dict[str, Any]) -> List[str]:
    result = []
    for item in article.get("subjects", []):
        if not isinstance(item, dict):
            continue
        value = str(item.get("fullName") or item.get("name") or "").strip()
        if value and value not in result:
            result.append(value)
    return result
