"""
Configuration loader for Krill project.
"""
import yaml
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class DatasetConfig(BaseModel):
    path: str
    split: str = Field(default="train")
    text_column: str = Field(default="text")


class DatatroveConfig(BaseModel):
    """Configuration for datatrove integration."""
    enabled: bool = Field(default=False, description="Enable datatrove preprocessing")
    deduplication_algorithm: str = Field(default="minhash", description="Deduplication algorithm: 'minhash' or 'exact'")
    quality_filters: Dict[str, Any] = Field(default_factory=lambda: {"min_length": 100}, description="Quality filter settings")
    distributed: bool = Field(default=False, description="Enable distributed processing")
    streaming: bool = Field(default=True, description="Enable streaming processing")
    num_workers: int = Field(default=1, description="Number of worker processes")
    minhash_threshold: float = Field(default=0.8, description="MinHash similarity threshold for deduplication")
    
    class Config:
        extra = "allow"  # Allow additional datatrove-specific settings


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
    muon_implementation: str = Field(default="moonlight")

    model_config_name: str = Field(default="small")
    gradient_accumulation_steps: int = Field(default=1)
    # The number of samples to include in each batch. This is the number of samples sent to
    # each GPU. Batch size per gpu = micro_batch_size * gradient_accumulation_steps
    micro_batch_size: int | None = Field(default=1)
    
    # Datatrove integration
    datatrove: Optional[DatatroveConfig] = Field(default_factory=lambda: DatatroveConfig(), description="Datatrove preprocessing configuration")


def load_config(path: str) -> KrillConfig:
    """Load YAML configuration and validate it against the Config model."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return KrillConfig(**data)
