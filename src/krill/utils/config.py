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


class KrillConfig(BaseModel):
    sequence_len: int
    vocab_size: int = Field(default=32000)
    hub_tokenizer_id: str
    dataset_prepared_path: str
    dataset_prepared_min_length: int = Field(default=150)
    datasets: List[DatasetConfig]
    hub_model_id: str
    output_dir: str
    num_epochs: int = Field(default=1)
    learning_rate: float = Field(default=3e-4)
    weight_decay: float = Field(default=0.01)
    optimizer: str = Field(default="muon")
    model_config_name: str = Field(default="small")
    gradient_accumulation_steps: int = Field(default=1)
    # The number of samples to include in each batch. This is the number of samples sent to
    # each GPU. Batch size per gpu = micro_batch_size * gradient_accumulation_steps
    micro_batch_size: int | None = Field(default=1)


def load_config(path: str) -> KrillConfig:
    """Load YAML configuration and validate it against the Config model."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return KrillConfig(**data)
