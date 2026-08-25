"""Framework-independent validation for the demo API request contract."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SearchRequest:
    query: str
    mode: str = "semantic"
    limit: int = 10
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    database: Optional[str] = None

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SearchRequest":
        return cls(
            query=str(value.get("query", "")),
            mode=str(value.get("mode", "semantic")),
            limit=int(value.get("limit", 10)),
            year_from=int(value["year_from"]) if value.get("year_from") not in (None, "") else None,
            year_to=int(value["year_to"]) if value.get("year_to") not in (None, "") else None,
            database=str(value["database"]) if value.get("database") else None,
        )
