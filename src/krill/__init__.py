import torch
from transformers.utils.import_utils import _is_package_available
from platform import system as platform_system


def get_device_type():
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    raise NotImplementedError("🦐 Krill: currently only works on NVIDIA GPUs.")


DEVICE_TYPE: str = get_device_type()

DEVICE_COUNT: int = torch.cuda.device_count()

PLATFORM_SYSTEM: str = platform_system()

SUPPORTS_BFLOAT16: bool = False

HAS_FLASH_ATTENTION: bool = False

try:
    major_version, minor_version = torch.cuda.get_device_capability()
    if major_version >= 8:
        SUPPORTS_BFLOAT16 = True
        if _is_package_available("flash_attn"):
            HAS_FLASH_ATTENTION = True
except Exception as e:
    raise ImportError(
        "🦐 Krill: Flash Attention 2 is not installed. Please install it to use this feature.")
