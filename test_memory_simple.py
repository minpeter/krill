#!/usr/bin/env python3
"""
Simple test to compare memory usage between standard and memory-efficient modes.
Uses very small dataset to avoid disk space issues.
"""

import os
import tempfile
import yaml
import psutil
from src.krill.utils.config import KrillConfig
from src.krill.utils.memory_monitor import MemoryMonitor
from src.krill.preprocess import _do_preprocess_standard, _do_preprocess_memory_efficient

def get_memory_mb():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def create_test_config(memory_efficient=False, output_dir=None):
    """Create a test configuration."""
    config_dict = {
        'vocab_size': 48000,
        'hub_tokenizer_id': 'openai-community/gpt2',  # Use a publicly available tokenizer
        'sequence_len': 2048,
        'dataset_prepared_path': output_dir or './test_output',
        'dataset_prepared_min_length': 100,
        'preprocess_memory_efficient': memory_efficient,
        'preprocess_chunk_size': 500,
        'preprocess_save_shard_size': '10MB',
        'hub_model_id': 'test/model',  # Required field
        'output_dir': output_dir or './test_output',  # Required field
        'datasets': [
            {
                'path': 'wikitext',
                'name': 'wikitext-2-raw-v1',
                'split': 'train[:100]',  # Very small for testing
                'text_column': 'text'
            }
        ]
    }
    return KrillConfig(**config_dict)

def test_memory_mode(mode_name, use_memory_efficient=False):
    """Test a specific mode and return peak memory increase."""
    print(f"\n{'='*50}")
    print(f"TESTING {mode_name.upper()} MODE")
    print(f"{'='*50}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            config = create_test_config(memory_efficient=use_memory_efficient, output_dir=temp_dir)
            monitor = MemoryMonitor()
            monitor.start_monitoring()
            
            if use_memory_efficient:
                _do_preprocess_memory_efficient(config, monitor)
            else:
                _do_preprocess_standard(config, monitor)
                
            peak_increase = monitor.peak_memory - monitor.start_memory
            print(f"✅ {mode_name} mode completed successfully")
            print(f"📊 Peak memory increase: {peak_increase:.1f} MB")
            return peak_increase
            
        except Exception as e:
            print(f"❌ {mode_name} mode failed: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == "__main__":
    print("🧠 Simple Memory Test for Krill Preprocessing")
    print("Using very small dataset to test memory efficiency...")
    
    # Test both modes
    standard_increase = test_memory_mode("Standard", use_memory_efficient=False)
    efficient_increase = test_memory_mode("Memory-Efficient", use_memory_efficient=True)
    
    # Compare results
    print(f"\n{'='*60}")
    print("MEMORY USAGE COMPARISON")
    print(f"{'='*60}")
    
    if standard_increase is not None:
        print(f"Standard mode peak increase:        {standard_increase:.1f} MB")
    else:
        print("Standard mode:                      FAILED")
        
    if efficient_increase is not None:
        print(f"Memory-efficient mode peak increase: {efficient_increase:.1f} MB")
    else:
        print("Memory-efficient mode:              FAILED")
    
    print(f"{'='*60}")
    
    if standard_increase is not None and efficient_increase is not None:
        if efficient_increase < standard_increase:
            savings = standard_increase - efficient_increase
            print(f"✅ Memory-efficient mode saves {savings:.1f} MB ({savings/standard_increase*100:.1f}%)")
        elif efficient_increase > standard_increase:
            overhead = efficient_increase - standard_increase
            print(f"❌ Memory-efficient mode uses {overhead:.1f} MB MORE than standard mode")
            print("   This indicates the memory-efficient implementation needs improvement.")
        else:
            print("🤷 Both modes use approximately the same amount of memory")
    else:
        print("❌ Cannot compare modes due to failures")