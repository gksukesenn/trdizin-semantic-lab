"""Tek kaynak JSON config yükleme ve doğrulama."""

import json
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_TOP = ["target_article_count", "paths", "api", "metadata_fields", "embedding", "umap", "hdbscan", "kmeans", "evaluation", "random_seed"]
REQUIRED_PATHS = ["dataset", "checkpoint", "pilot_dataset", "validation_dataset", "embedding", "embedding_metadata", "embedding_progress", "output_root"]
REQUIRED_API = ["endpoint", "timeout_connect_seconds", "timeout_read_seconds", "retry_count", "request_delay_seconds", "page_limit", "pages_per_query", "document_type", "publication_language", "require_abstract", "years", "queries"]
REQUIRED_EMBEDDING = ["model_id", "max_seq_length", "expected_dimension", "normalize_embeddings", "require_cuda", "batch_size_candidates", "chunk_size"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _missing(mapping: Dict[str, Any], names: List[str], section: str) -> None:
    absent = [name for name in names if name not in mapping]
    if absent:
        raise ValueError("Config eksik alan (%s): %s" % (section, ", ".join(absent)))


def load_config(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except FileNotFoundError as error:
        raise FileNotFoundError("Config bulunamadı: %s" % path) from error
    except json.JSONDecodeError as error:
        raise ValueError("Config geçerli JSON değil: %s" % path) from error
    if not isinstance(config, dict):
        raise ValueError("Config kökü JSON nesnesi olmalıdır.")
    _missing(config, REQUIRED_TOP, "kök")
    for name, required in [("paths", REQUIRED_PATHS), ("api", REQUIRED_API), ("embedding", REQUIRED_EMBEDDING)]:
        if not isinstance(config[name], dict):
            raise ValueError("Config '%s' alanı nesne olmalıdır." % name)
        _missing(config[name], required, name)
    if int(config["target_article_count"]) <= 0:
        raise ValueError("target_article_count pozitif olmalıdır.")
    if config["api"]["document_type"] != "PAPER" or config["api"]["publication_language"] != "TUR":
        raise ValueError("Nihai pipeline document_type=PAPER ve publication_language=TUR gerektirir.")
    if int(config["embedding"]["expected_dimension"]) != 768:
        raise ValueError("TR-MTEB expected_dimension 768 olmalıdır.")
    config["_config_path"] = str(path)
    config["_project_root"] = str(project_root())
    return config


def resolve_path(config: Dict[str, Any], key: str) -> Path:
    raw = Path(str(config["paths"][key]))
    return raw if raw.is_absolute() else project_root() / raw


def ensure_output_directories(config: Dict[str, Any]) -> None:
    root = resolve_path(config, "output_root")
    for name in ["embeddings", "clustering", "figures", "reports", "logs", "search"]:
        (root / name).mkdir(parents=True, exist_ok=True)
    resolve_path(config, "checkpoint").parent.mkdir(parents=True, exist_ok=True)
