import os
import shutil
from typing import Iterable, Optional

import torch


def set_visible_device(device_index: int) -> None:
    device_value = str(device_index)
    os.environ["CUDA_VISIBLE_DEVICES"] = device_value
    os.environ["HIP_VISIBLE_DEVICES"] = device_value


def synchronize_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_profiler_gpu_activity() -> Optional[torch.profiler.ProfilerActivity]:
    if not torch.cuda.is_available():
        return None
    return torch.profiler.ProfilerActivity.CUDA


def get_runtime_trace_categories() -> set[str]:
    return {
        "cuda_runtime",
        "hip_runtime",
        "rocm_runtime",
    }


def get_total_gpu_memory_bytes(device_index: int = 0) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("No accelerator device available")

    current_index = torch.cuda.current_device()
    if current_index != device_index:
        torch.cuda.set_device(device_index)

    try:
        free_memory, total_memory = torch.cuda.mem_get_info()
        del free_memory
        return int(total_memory)
    except Exception:
        props = torch.cuda.get_device_properties(device_index)
        return int(props.total_memory)


def resolve_runtime_gpu_vendor(gpu_vendor: str = "auto") -> str:
    normalized_vendor = gpu_vendor.lower()
    if normalized_vendor in {"nvidia", "amd"}:
        return normalized_vendor
    if normalized_vendor != "auto":
        raise ValueError(f"Unsupported gpu vendor: {gpu_vendor}")

    if getattr(torch.version, "hip", None):
        return "amd"
    if getattr(torch.version, "cuda", None):
        return "nvidia"

    has_amd = shutil.which("amd-smi") is not None
    has_nvidia = shutil.which("nvidia-smi") is not None
    if has_amd and not has_nvidia:
        return "amd"
    if has_nvidia:
        return "nvidia"
    if has_amd:
        return "amd"
    return "nvidia"


def resolve_attention_backend_for_vendor(
    attention_backend: str,
    available_backends: Iterable[str],
    gpu_vendor: str,
) -> str:
    resolved_vendor = resolve_runtime_gpu_vendor(gpu_vendor)
    normalized_backend = attention_backend.lower()
    if resolved_vendor != "amd" or "flashinfer" not in normalized_backend:
        return attention_backend

    fallback_candidates = [
        "triton",
        "xformers",
        "flash_attn",
        "sdpa",
        "torch_sdpa",
    ]
    backend_by_name = {backend.lower(): backend for backend in available_backends}

    for candidate in fallback_candidates:
        if candidate in backend_by_name:
            return backend_by_name[candidate]

    raise RuntimeError(
        "FlashInfer backend is selected, but no ROCm-compatible attention backend is available. "
        "Pass --attention_backend explicitly to a backend supported by your runtime build."
    )
