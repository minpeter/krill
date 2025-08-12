import os
import importlib.util
from platform import system as platform_system

__version__ = "2025.8.2"


def _is_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def get_device_type() -> str:
    """Return 'cuda' if a CUDA-enabled GPU is available, else 'cpu'.
    Avoids importing heavy modules at package import time.
    """
    try:
        import torch  # Imported lazily

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def device_count() -> int:
    """Return number of CUDA devices if available; 0 otherwise."""
    try:
        import torch  # Imported lazily

        return torch.cuda.device_count()
    except Exception:
        return 0


def supports_bfloat16() -> bool:
    """Best-effort detection for bfloat16 support without hard-failing."""
    try:
        import torch  # Imported lazily

        if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
            return False
        major_version, _ = torch.cuda.get_device_capability()
        return major_version >= 8
    except Exception:
        return False


def has_flash_attention() -> bool:
    """Detect availability of FlashAttention-2 in compatible environments."""
    try:
        return supports_bfloat16() and _is_installed("flash_attn")
    except Exception:
        return False


def get_statistics() -> str:
    """Return environment statistics string for Krill with lazy imports."""
    device_type = get_device_type()
    num_devices = device_count()
    platform_name = platform_system()

    if device_type == "cuda":
        try:
            import torch  # Imported lazily

            gpu_props = torch.cuda.get_device_properties(0)
            gpu_stats_snippet = (
                f"CUDA: {gpu_props.major}.{gpu_props.minor}. "
                f"CUDA Toolkit: {getattr(torch.version, 'cuda', 'N/A')}."
            )
            max_mem = round(gpu_props.total_memory / 1024 / 1024 / 1024, 3)
            device_name = gpu_props.name
            extra = ""
        except Exception:
            gpu_stats_snippet = "GPU detected, but failed to query properties."
            max_mem = "N/A"
            device_name = "CUDA"
            extra = ""
    else:
        gpu_stats_snippet = "No GPU detected. Running on CPU."
        max_mem = "N/A"
        device_name = "CPU"
        extra = f" CPU cores: {os.cpu_count() or 'N/A'}."

    triton_version = "N/A"
    if _is_installed("triton"):
        try:
            import triton  # Imported lazily

            triton_version = triton.__version__
        except Exception:
            pass

    transformers_version = "N/A"
    if _is_installed("transformers"):
        try:
            from transformers import __version__ as _transformers_version  # Imported lazily

            transformers_version = _transformers_version
        except Exception:
            pass

    try:
        import torch  # Imported lazily

        torch_version = torch.__version__
    except Exception:
        torch_version = "N/A"

    stats = (
        f"  /¯¯¯¯\\   🦐 Krill {__version__}: A minimal pretraining trainer for LLMs — from scratch.\n"
        f" ( #|\\_ü|  {device_name}. Num GPUs = {num_devices}. Max memory: {max_mem} GB. Platform: {platform_name}.{extra}\n"
        f" ( #\\  ƒƒ  Torch: {torch_version}. {gpu_stats_snippet} Triton: {triton_version}\n"
        f"  \\ #\\     Transformers: {transformers_version}. Bfloat16 = {supports_bfloat16()}. FA2 = {has_flash_attention()}\n"
        f"  /|||\\    Source code: https://github.com/minpeter/krill\n"
    )
    return stats
