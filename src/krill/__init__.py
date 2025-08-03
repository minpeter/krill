import torch
from platform import system as platform_system
from transformers.utils.import_utils import _is_package_available
from transformers import __version__ as transformers_version
from triton import __version__ as triton_version

__version__ = "2025.8.2"


def get_device_type():
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


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
except Exception:
    pass


if DEVICE_TYPE == "cuda":
    gpu_stats = torch.cuda.get_device_properties(0)
    gpu_version = torch.version.cuda
    gpu_stats_snippet = f"CUDA: {gpu_stats.major}.{gpu_stats.minor}. CUDA Toolkit: {gpu_version}."
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    gpu_name = gpu_stats.name
else:
    gpu_stats = torch.device("cpu")
    gpu_stats_snippet = "No GPU detected. Running on CPU."
    max_memory = "N/A"
    gpu_name = "CPU"

statistics = \
    f"  /¯¯¯¯{chr(92)}   🦐 Krill {__version__}: A minimal pretraining trainer for LLMs — from scratch.\n"\
    f" ( #|{chr(92)}_ü|  {gpu_name}. \033[1;43;30mNum GPUs = {DEVICE_COUNT}.\033[0m Max memory: {max_memory} GB. Platform: {PLATFORM_SYSTEM}.\n"\
    f" ( #{chr(92)}  ƒƒ  Torch: {torch.__version__}. {gpu_stats_snippet} Triton: {triton_version}\n"\
    f"  {chr(92)} #{chr(92)}     Transformers: {transformers_version}. Bfloat16 = {SUPPORTS_BFLOAT16}. FA2 = {HAS_FLASH_ATTENTION}\n"\
    f'  /|||{chr(92)}    Source code: https://github.com/minpeter/krill\n'

print(statistics)
