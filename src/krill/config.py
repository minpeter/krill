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

class Config(BaseModel):
    sequence_len: int
    hub_tokenizer_id: str
    dataset_prepared_path: str
    dataset_prepared_min_length: Optional[int]
    datasets: List[DatasetConfig]
    hub_model_id: Optional[str]
    output_dir: Optional[str]
    num_epochs: Optional[int]
    learning_rate: Optional[float]
    weight_decay: Optional[float]
    optimizer: Optional[str]
    model_config_name: Optional[str]

def load_config(path: str) -> Config:
    """Load YAML configuration and validate it against the Config model."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return Config(**data)
