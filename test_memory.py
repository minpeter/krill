#!/usr/bin/env python3
"""
Quick test script to compare memory usage between standard and memory-efficient modes.
"""
import os
import shutil
import subprocess
import time

def run_test(mode: bool, test_name: str):
    """Run preprocessing test with given memory efficiency mode."""
    print(f"\n{'='*60}")
    print(f"Testing {test_name} mode (preprocess_memory_efficient: {mode})")
    print(f"{'='*60}")
    
    # Clean up artifacts
    artifacts_dir = "./artifacts"
    if os.path.exists(artifacts_dir):
        shutil.rmtree(artifacts_dir)
    
    # Create test config
    config_content = f"""# Test configuration for memory comparison
vocab_size: 32000
hub_tokenizer_id: minpeter/webtext-tokenizer-32k

sequence_len: 2048
dataset_prepared_path: ./artifacts/dataset_test
dataset_prepared_min_length: 100

# Memory efficiency settings
preprocess_memory_efficient: {str(mode).lower()}
preprocess_chunk_size: 2000
preprocess_dedup_cache_dir: ./artifacts/cache/dedup
preprocess_save_shard_size: 50MB

# Small dataset for testing
datasets:
  - path: wikitext
    name: wikitext-2-raw-v1
    split: train
    text_column: text

# Training config (not used)
hub_model_id: test/model
output_dir: ./artifacts/models/test_model
num_epochs: 1
learning_rate: 3e-4
weight_decay: 0.01
optimizer: muon
micro_batch_size: 4
gradient_accumulation_steps: 8
model_config_name: small
"""
    
    with open("test_config.yaml", "w") as f:
        f.write(config_content)
    
    # Run preprocessing
    cmd = ["python", "-m", "krill.main", "preprocess", "test_config.yaml"]
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        print(f"Exit code: {result.returncode}")
        print(f"Duration: {end_time - start_time:.2f} seconds")
        
        if result.returncode == 0:
            print("✅ Success!")
            # Print last few lines of output to see memory summary
            output_lines = result.stdout.strip().split('\n')
            print("\nLast 10 lines of output:")
            for line in output_lines[-10:]:
                print(f"  {line}")
        else:
            print("❌ Failed!")
            print("STDOUT:", result.stdout[-1000:])  # Last 1000 chars
            print("STDERR:", result.stderr[-1000:])  # Last 1000 chars
            
    except subprocess.TimeoutExpired:
        print("❌ Timeout after 5 minutes")
    
    # Clean up
    if os.path.exists("test_config.yaml"):
        os.remove("test_config.yaml")
    if os.path.exists(artifacts_dir):
        shutil.rmtree(artifacts_dir)
    
    print(f"{'='*60}")

if __name__ == "__main__":
    os.chdir("/home/runner/work/krill/krill")
    
    # Test both modes
    run_test(False, "Standard")
    time.sleep(5)  # Brief pause between tests
    run_test(True, "Memory-Efficient")