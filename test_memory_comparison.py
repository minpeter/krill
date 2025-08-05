#!/usr/bin/env python3
"""
Memory comparison test between standard and memory-efficient preprocessing.
"""

import os
import shutil
import tempfile
from pathlib import Path

from krill.utils.config import load_config
from krill.preprocess import do_preprocess


def run_memory_comparison_test():
    """Run a comparison test between standard and memory-efficient modes."""
    print("🧪 Memory Efficiency Comparison Test")
    print("=" * 50)
    
    # Create temporary test configs
    base_config = {
        'vocab_size': 32000,
        'hub_tokenizer_id': 'minpeter/webtext-tokenizer-32k',
        'sequence_len': 512,
        'dataset_prepared_min_length': 50,
        'datasets': [
            {
                'path': 'HAERAE-HUB/KOREAN-WEBTEXT',
                'split': 'train[:500]',  # Small subset for testing
                'text_column': 'text'
            }
        ],
        'hub_model_id': 'test/memory-test',
        'output_dir': './test_output',
        'num_epochs': 1,
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'optimizer': 'muon',
        'muon_implementation': 'moonlight',
        'micro_batch_size': 1,
        'gradient_accumulation_steps': 1,
        'model_config_name': 'pico'
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test 1: Standard mode
        print("\n📚 Testing Standard Mode")
        print("-" * 30)
        
        standard_config = base_config.copy()
        standard_config.update({
            'dataset_prepared_path': str(temp_path / 'standard_output'),
            'preprocess_memory_efficient': False
        })
        
        # Save config file
        import yaml
        standard_config_path = temp_path / 'standard_config.yaml'
        with open(standard_config_path, 'w') as f:
            yaml.dump(standard_config, f)
        
        # Load and process
        config = load_config(str(standard_config_path))
        do_preprocess(config)
        
        print("\n🧠 Testing Memory-Efficient Mode")
        print("-" * 30)
        
        # Test 2: Memory-efficient mode
        memory_config = base_config.copy()
        memory_config.update({
            'dataset_prepared_path': str(temp_path / 'memory_output'),
            'preprocess_memory_efficient': True,
            'preprocess_chunk_size': 100,  # Small chunks for testing
            'preprocess_dedup_cache_dir': str(temp_path / 'dedup_cache')
        })
        
        # Save config file
        memory_config_path = temp_path / 'memory_config.yaml'
        with open(memory_config_path, 'w') as f:
            yaml.dump(memory_config, f)
        
        # Load and process
        config = load_config(str(memory_config_path))
        do_preprocess(config)
        
        print("\n✅ Memory comparison test completed!")
        print("Both modes should produce similar results with different memory patterns.")


if __name__ == "__main__":
    run_memory_comparison_test()