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
    
    # Native datatrove preprocessing options
    deduplication_algorithm: str = Field(default="minhash", description="Deduplication algorithm: 'minhash' or 'exact'")
    min_length: int = Field(default=100, description="Minimum text length for quality filtering")
    max_length: Optional[int] = Field(default=None, description="Maximum text length for quality filtering")
    use_trafilatura: bool = Field(default=False, description="Use Trafilatura for text extraction")
    collect_stats: bool = Field(default=True, description="Collect statistics during processing")
    cleanup_temp: bool = Field(default=True, description="Clean up temporary files after processing")
    streaming: bool = Field(default=True, description="Enable streaming processing")
    num_workers: int = Field(default=1, description="Number of worker processes")
    minhash_threshold: float = Field(default=0.8, description="MinHash similarity threshold for deduplication")
    
    # Optional HuggingFace dataset upload
    dataset_prepared_hf_id: Optional[str] = Field(default=None, description="HuggingFace dataset ID for uploading processed dataset (e.g., 'username/dataset-name')")


def load_config(path: str) -> KrillConfig:
    """Load YAML configuration and validate it against the Config model."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return KrillConfig(**data)
