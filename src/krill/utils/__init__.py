import os
from typing import Optional
import os.path as _osp
import yaml


def patch_optimized_env():
    """
    Patch environment variables to improve VRAM usage and increase download speed
    """
    if os.getenv("HF_HUB_ENABLE_HF_TRANSFER") is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def resolve_model_arg(model: str) -> str:
    """Resolve a CLI model argument into a Hugging Face model id.

    Behavior:
    - If `model` looks like a YAML file path and exists, load it and return train.hub_model_id.
    - Otherwise, treat `model` as already a HF model id and return as-is.

    Raises:
    - FileNotFoundError: if YAML path given but not found
    - ValueError: if YAML file doesn't contain train.hub_model_id
    - yaml.YAMLError: if YAML parsing fails
    """
    if _osp.isfile(model) and model.lower().endswith((".yaml", ".yml")):
        print(f"⚓️ Loading config file: {model}...")
        with open(model, "r") as f:
            cfg = yaml.safe_load(f) or {}
        hub_model_id: Optional[str] = (
            (cfg.get("train") or {}).get("hub_model_id")
        )
        if not hub_model_id:
            raise ValueError(
                "hub_model_id not found in config under 'train.hub_model_id'")
        print(f"⚓️ Using model from config: {hub_model_id}...")
        return hub_model_id
    return model
