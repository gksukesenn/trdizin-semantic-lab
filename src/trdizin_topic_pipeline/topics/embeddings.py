"""CUDA doğrulamalı, resume destekli TR-MTEB embedding üretimi."""

import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ..utils.io import atomic_json, file_sha256


def duplicate_vector_count(values: np.ndarray) -> int:
    contiguous = np.ascontiguousarray(values)
    packed = contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))).reshape(-1)
    return int(len(packed) - len(np.unique(packed)))


def select_device(allow_cpu: bool) -> Tuple[Any, str, Dict[str, Any]]:
    import torch
    available = bool(torch.cuda.is_available())
    device = "cuda:0" if available else "cpu"
    facts = {"python_executable": sys.executable, "python_version": platform.python_version(),
             "torch_version": torch.__version__, "torch_cuda_build": torch.version.cuda,
             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<ayarlanmamış>"),
             "torch_cuda_available": available, "cuda_device_count": int(torch.cuda.device_count()),
             "selected_device": device,
             "gpu_name": torch.cuda.get_device_name(0) if available else None}
    for label, value in [("Python executable", facts["python_executable"]), ("Python version", facts["python_version"]),
                         ("torch version", facts["torch_version"]), ("torch CUDA build", facts["torch_cuda_build"]),
                         ("CUDA_VISIBLE_DEVICES", facts["cuda_visible_devices"]),
                         ("torch.cuda.is_available()", facts["torch_cuda_available"]),
                         ("Device count", facts["cuda_device_count"]), ("Seçilen device", device),
                         ("GPU adı", facts["gpu_name"])]:
        print("%-27s: %s" % (label, value))
    if not available and not allow_cpu:
        raise RuntimeError("CUDA kullanılamıyor. Sessiz CPU fallback yapılmadı; CPU için açıkça --allow-cpu verin.")
    return torch, device, facts


def print_device_facts(facts: Dict[str, Any], batch_size: int, model_source: str,
                       model_device: str, parameter_device: str) -> None:
    labels = [("Python executable", facts["python_executable"]), ("Python version", facts["python_version"]),
              ("torch version", facts["torch_version"]), ("torch CUDA build", facts["torch_cuda_build"]),
              ("CUDA_VISIBLE_DEVICES", facts["cuda_visible_devices"]),
              ("torch.cuda.is_available()", facts["torch_cuda_available"]),
              ("Device count", facts["cuda_device_count"]), ("Seçilen device", facts["selected_device"]),
              ("GPU adı", facts["gpu_name"]), ("Batch size", batch_size), ("Model source", model_source),
              ("model.device", model_device), ("İlk parametre device", parameter_device)]
    for label, value in labels:
        print("%-27s: %s" % (label, value))


def _devices_match(selected: str, actual: str) -> bool:
    return actual.startswith("cuda") if selected.startswith("cuda") else actual == "cpu"


def build_embeddings(rows: List[Dict[str, Any]], dataset_path: Path, output: Path,
                     metadata_path: Path, progress_path: Path, settings: Dict[str, Any],
                     allow_cpu: bool = False, limit: int = 0) -> Dict[str, Any]:
    torch, device, facts = select_device(allow_cpu)
    from sentence_transformers import SentenceTransformer
    selected_rows = rows[:limit] if limit else rows
    count, dimension = len(selected_rows), int(settings["expected_dimension"])
    dataset_hash = file_sha256(dataset_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    progress: Dict[str, Any] = {}
    if progress_path.exists():
        import json
        with progress_path.open("r", encoding="utf-8") as handle: progress = json.load(handle)
    completed = int(progress.get("completed_rows", 0)) if progress.get("dataset_sha256") == dataset_hash else 0
    mode = "r+" if output.exists() and completed else "w+"
    values = np.lib.format.open_memmap(str(output), mode=mode, dtype=np.float32, shape=(count, dimension))
    model = SentenceTransformer(str(settings["model_id"]), device=device, trust_remote_code=False)
    model.max_seq_length = int(settings["max_seq_length"])
    model_device = str(model.device)
    parameter_device = str(next(model.parameters()).device)
    if not _devices_match(device, model_device) or not _devices_match(device, parameter_device):
        raise RuntimeError("Model cihaz uyuşmazlığı: seçilen=%s model=%s parametre=%s" % (device, model_device, parameter_device))
    batch_candidates = [int(value) for value in settings["batch_size_candidates"]]
    if device == "cpu": batch_candidates = [min(16, value) for value in batch_candidates]
    batch_index = 0
    batch_size = batch_candidates[batch_index]
    print_device_facts(facts, batch_size, str(settings["model_id"]), model_device, parameter_device)
    started = time.time()
    chunk_size = int(settings["chunk_size"])
    while completed < count:
        end = min(completed + chunk_size, count)
        texts = [str(row["abstract_tr"]) for row in selected_rows[completed:end]]
        try:
            encoded = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                                   normalize_embeddings=bool(settings["normalize_embeddings"]),
                                   show_progress_bar=True).astype(np.float32, copy=False)
        except RuntimeError as error:
            is_oom = "out of memory" in str(error).lower()
            if not is_oom or batch_index + 1 >= len(batch_candidates): raise
            batch_index += 1; batch_size = batch_candidates[batch_index]
            if device.startswith("cuda"): torch.cuda.empty_cache()
            print("CUDA OOM; batch size %d olarak yeniden deneniyor." % batch_size)
            continue
        if encoded.shape != (end - completed, dimension):
            raise ValueError("Beklenmeyen embedding şekli: %s" % (encoded.shape,))
        values[completed:end] = encoded; values.flush(); completed = end
        progress = {"dataset_sha256": dataset_hash, "completed_rows": completed, "total_rows": count,
                    "expected_dimension": dimension, "batch_size": batch_size, "complete": False,
                    "updated_at_epoch": time.time()}
        atomic_json(progress_path, progress)
        print("Embedding ilerleme          : %d/%d | geçen %.1f sn" % (completed, count, time.time() - started))
    final = np.load(str(output), mmap_mode="r")
    norms = np.linalg.norm(final, axis=1)
    if final.shape != (count, dimension) or final.dtype != np.float32 or not np.isfinite(final).all():
        raise ValueError("Nihai embedding shape/dtype/sonluluk doğrulaması başarısız.")
    if not np.allclose(norms, 1.0, atol=1e-4): raise ValueError("Nihai embedding normları yaklaşık 1.0 değil.")
    max_memory = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
    metadata = dict(facts)
    metadata.update({"dataset_sha256": dataset_hash, "row_count": count, "shape": [count, dimension],
                     "dtype": "float32", "model_id": settings["model_id"], "model_device": model_device,
                     "first_parameter_device": parameter_device, "actual_device": device,
                     "batch_size": batch_size, "max_seq_length": settings["max_seq_length"],
                     "normalize_embeddings": True, "duplicate_vector_count": duplicate_vector_count(final),
                     "torch_cuda_max_memory_allocated": max_memory, "row_alignment": "JSONL sıra indeksi == embedding sıra indeksi"})
    atomic_json(metadata_path, metadata)
    progress["complete"] = True; atomic_json(progress_path, progress)
    print("torch.cuda.max_memory_allocated(): %d" % max_memory)
    return metadata
