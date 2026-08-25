"""Shared query model/device lifecycle and 768D encoding validation."""

import threading
import time
from typing import Any, Tuple

import numpy as np


def select_device(allow_cpu: bool, torch_module: Any = None) -> str:
    if torch_module is None:
        import torch as torch_module
    if torch_module.cuda.is_available():
        return "cuda"
    if allow_cpu:
        return "cpu"
    raise RuntimeError("CUDA bulunamadı. CPU kullanmak için --allow-cpu verin.")


def load_model(model_id: str, device: str, max_seq_length: int = 512) -> Any:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device=device, trust_remote_code=False)
    model.max_seq_length = max_seq_length
    return model


class QueryEncoder:
    """Owns one embedding model and serializes thread-safe query encoding."""

    def __init__(self, model_id: str, allow_cpu: bool, model: Any = None, torch_module: Any = None) -> None:
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.device = select_device(allow_cpu, torch_module)
        self.model = model if model is not None else load_model(model_id, self.device)
        self.model_lock = threading.Lock()

    def encode(self, query: str) -> Tuple[np.ndarray, float]:
        started = time.perf_counter()
        with self.model_lock:
            values = self.model.encode(
                [query], batch_size=1, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            )
            if self.device == "cuda":
                self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        values = values.astype(np.float32, copy=False)
        if values.shape != (1, 768):
            raise RuntimeError("Sorgu embedding şekli yanlış: %r" % (values.shape,))
        if not np.isfinite(values).all():
            raise RuntimeError("Sorgu embeddingi NaN/Inf içeriyor.")
        return values[0], elapsed
