"""Shared model, path and terminal-format helpers for search commands."""

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from .query_encoder import load_model as _load_model
from ..utils.io import read_json


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def select_cli_device(allow_cpu: bool, error_message: str = "CUDA bulunamadı. CPU kullanımı için --allow-cpu parametresini verin.") -> str:
    if torch.cuda.is_available():
        return "cuda"
    if allow_cpu:
        return "cpu"
    raise RuntimeError(error_message)


def load_cli_model(model_id: str, device: str, style: str = "standard") -> Any:
    model = _load_model(model_id, device)
    parameter_device = next(model.parameters()).device
    if style == "three_way":
        print("model.device       :", model.device)
        print("İlk parametre      :", parameter_device)
    elif style == "benchmark":
        print("Model device          :", model.device)
        print("İlk parametre device  :", parameter_device)
    else:
        print("Model                     :", model_id)
        print("model.device              :", model.device)
        print("İlk parametre cihazı      :", parameter_device)
    if str(model.device).split(":")[0] != device:
        raise RuntimeError("Model beklenen cihazda değil.")
    if str(parameter_device).split(":")[0] != device:
        raise RuntimeError("Model parametreleri beklenen cihazda değil.")
    return model


def encode_queries(model: Any, query_texts: Any, device: str) -> np.ndarray:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    values = model.encode(
        query_texts, batch_size=min(16, len(query_texts)), convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    values = values.astype(np.float32, copy=False)
    expected_shape = (len(query_texts), 768)
    if values.shape != expected_shape:
        raise RuntimeError("Sorgu embedding şekli yanlış: %r" % (values.shape,))
    if not np.isfinite(values).all():
        raise RuntimeError("Sorgu embeddingleri NaN/Inf içeriyor.")
    if not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-4):
        raise RuntimeError("Sorgu embeddingleri normalize değil.")
    print("Sorgu embedding süresi: %.4f sn" % elapsed)
    if device == "cuda":
        print("Tepe CUDA belleği    : %.2f MiB" % (torch.cuda.max_memory_allocated() / 1024**2))
    return values


def encode_cli_query(model: Any, query: str, device: str, style: str = "standard") -> np.ndarray:
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    values = model.encode(
        [query], batch_size=1, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    values = values.astype(np.float32, copy=False)
    if values.shape != (1, 768):
        raise RuntimeError("Sorgu embedding şekli yanlış: %r" % (values.shape,))
    if not np.isfinite(values).all():
        raise RuntimeError("Sorgu embeddingi NaN/Inf içeriyor.")
    norm = float(np.linalg.norm(values[0]))
    if not np.isclose(norm, 1.0, atol=1e-4):
        raise RuntimeError("Sorgu embeddingi normalize değil.")
    if style == "three_way":
        print("Sorgu embedding süresi : %.4f sn" % elapsed)
        print("Embedding normu        : %.6f" % norm)
    else:
        print("Sorgu embedding süresi    : %.4f sn" % elapsed)
        print("Sorgu embedding normu     : %.6f" % norm)
        if device == "cuda":
            print("Tepe CUDA belleği         : %.2f MiB" % (torch.cuda.max_memory_allocated() / 1024**2))
    return values[0]


def shorten(value: Any, maximum_length: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= maximum_length else text[: maximum_length - 3] + "..."


def value_or_dash(value: Any) -> str:
    return str(value) if value is not None else "—"


def score_or_dash(value: Any) -> str:
    return "%.4f" % float(value) if value is not None else "—"
