import os


def patch_optimized_env():
    """
    Patch environment variables to improve VRAM usage and increase download speed
    """
    if os.getenv("HF_HUB_ENABLE_HF_TRANSFER") is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
