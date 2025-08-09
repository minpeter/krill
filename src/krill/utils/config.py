"""
Configuration loader for Krill project.
"""
import yaml
from pydantic import BaseModel, Field
from typing import List, Optional


class DatasetConfig(BaseModel):
    path: str
    split: str = Field(default="train")
    text_column: str = Field(default="text")


class TokenizerConfig(BaseModel):
    hub_id: str
    vocab_size: int = Field(default=32000)


class PreprocessConfig(BaseModel):
    prepared_path: str
    sequence_len: int
    min_length: int = Field(default=150)


class TrainConfig(BaseModel):
    hub_model_id: str
    output_dir: str
    num_epochs: int = Field(default=1)
    learning_rate: float = Field(default=3e-4)
    weight_decay: float = Field(default=0.01)
    optimizer: str = Field(default="muon")
    muon_implementation: str = Field(default="moonlight")
    # Model family/architecture, e.g., 'llama' or 'qwen3'
    arch: str = Field(default="llama")
    model_config_name: str = Field(default="small")
    gradient_accumulation_steps: int = Field(default=1)
    micro_batch_size: int | None = Field(default=1)


class KrillConfig(BaseModel):
    datasets: List[DatasetConfig]
    tokenizer: TokenizerConfig
    preprocess: PreprocessConfig
    train: TrainConfig


def load_config(path: str) -> KrillConfig:
    """Load YAML configuration and validate it against the Config model."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return KrillConfig(**data)
