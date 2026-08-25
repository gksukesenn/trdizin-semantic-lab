"""Day29 ile kanıtlanmış TR Dizin arama istemcisi."""

import time
from typing import Any, Dict, List, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def build_session(retry_count: int) -> requests.Session:
    retry = Retry(total=retry_count, connect=retry_count, read=retry_count,
                  backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset(["GET"]), raise_on_status=False)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"Accept": "application/json", "User-Agent": "trdizin-semantic-lab/1.0"})
    return session


def request_tasks(years: List[int], queries: List[str], pages_per_query: int) -> List[Tuple[int, str, int]]:
    tasks: List[Tuple[int, str, int]] = []
    for offset in range(len(queries)):
        for page in range(1, pages_per_query + 1):
            for year_position, year in enumerate(years):
                tasks.append((int(year), queries[(offset + year_position) % len(queries)], page))
    return tasks


def fetch_page(session: requests.Session, api: Dict[str, Any], year: int, query: str, page: int) -> Dict[str, Any]:
    params = {"q": query, "order": "publicationYear-DESC", "page": page,
              "limit": int(api["page_limit"]), "facet-documentType": "PAPER",
              "facet-publicationLanguage": "TUR", "facet-publication_year": year}
    response = session.get(str(api["endpoint"]), params=params,
                           timeout=(float(api["timeout_connect_seconds"]), float(api["timeout_read_seconds"])))
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError("TR Dizin yanıtı JSON nesnesi değil.")
    time.sleep(float(api["request_delay_seconds"]))
    return result
