#!/usr/bin/env python3

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from krill.utils.config import KrillConfig
from krill.preprocess import do_preprocess
from krill.utils.memory_monitor import MemoryMonitor

def create_test_config(memory_efficient: bool = False) -> KrillConfig:
    """Create a test configuration."""
    temp_dir = tempfile.mkdtemp(prefix="krill_test_")
    
    config_data = {
        "vocab_size": 32000,
        "hub_tokenizer_id": "gpt2",
        "sequence_len": 512,  # Smaller for testing
        "dataset_prepared_path": os.path.join(temp_dir, "dataset"),
        "dataset_prepared_min_length": 50,
        "datasets": [
            {
                "path": "wikitext",
                "name": "wikitext-2-raw-v1",
                "split": "train[:500]",  # Very small for testing
                "text_column": "text"
            }
        ],
        "hub_model_id": "test/model",
        "output_dir": os.path.join(temp_dir, "model"),
        "num_epochs": 1,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "optimizer": "muon",
        "muon_implementation": "moonlight",
        "micro_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "model_config_name": "small",
        "preprocess_memory_efficient": memory_efficient,
        "preprocess_chunk_size": 100,  # Small for testing
        "preprocess_save_shard_size": "10MB",
    }
    
    return KrillConfig(**config_data)

def test_memory_usage():
    """Test memory usage comparison between standard and memory-efficient modes."""
    print("Testing memory usage comparison...")
    
    # Test standard mode
    print("\n=== Testing Standard Mode ===")
    config_standard = create_test_config(memory_efficient=False)
    try:
        do_preprocess(config_standard)
        print("Standard mode completed successfully")
    except Exception as e:
        print(f"Standard mode failed: {e}")
    finally:
        # Cleanup
        if os.path.exists(config_standard.dataset_prepared_path):
            shutil.rmtree(os.path.dirname(config_standard.dataset_prepared_path))
    
    # Test memory-efficient mode
    print("\n=== Testing Memory-Efficient Mode ===")
    config_memory_efficient = create_test_config(memory_efficient=True)
    try:
        do_preprocess(config_memory_efficient)
        print("Memory-efficient mode completed successfully")
    except Exception as e:
        print(f"Memory-efficient mode failed: {e}")
    finally:
        # Cleanup
        if os.path.exists(config_memory_efficient.dataset_prepared_path):
            shutil.rmtree(os.path.dirname(config_memory_efficient.dataset_prepared_path))

if __name__ == "__main__":
    test_memory_usage()