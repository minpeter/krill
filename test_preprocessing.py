#!/usr/bin/env python3
"""
Comprehensive test script for preprocessing modes.
Tests both standard and memory-efficient modes and compares:
1. Memory usage (efficient should be less than standard)
2. Drop tokens and results consistency
3. Correctness of outputs
"""

import os
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path
import yaml

def create_test_config(preprocess_memory_efficient: bool, output_dir: str) -> str:
    """Create a test configuration file."""
    config = {
        'vocab_size': 48000,
        'hub_tokenizer_id': 'pretraining/fw2-edu-kr-tokenizer-48k',
        'sequence_len': 2048,
        'dataset_prepared_path': output_dir,
        'dataset_prepared_min_length': 100,
        'preprocess_memory_efficient': preprocess_memory_efficient,
        'preprocess_chunk_size': 500,
        'datasets': [{
            'path': 'wikitext',
            'name': 'wikitext-2-raw-v1', 
            'split': 'train[:1_000]',  # Small dataset for testing
            'text_column': 'text'
        }],
        # Required but not used for preprocessing
        'hub_model_id': 'test/model',
        'output_dir': './artifacts/models/test',
        'num_epochs': 1,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'optimizer': 'muon',
        'muon_implementation': 'moonlight',
        'micro_batch_size': 4,
        'gradient_accumulation_steps': 8,
        'model_config_name': 'small'
    }
    
    config_path = f"/tmp/test_config_{'' if preprocess_memory_efficient else 'standard'}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return config_path

def run_preprocessing(config_path: str) -> tuple[str, int]:
    """Run preprocessing and capture output and return code."""
    try:
        result = subprocess.run([
            'uv', 'run', '-m', 'krill.main', 'preprocess', config_path
        ], capture_output=True, text=True, timeout=300, cwd='/home/runner/work/krill/krill')
        return result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", 124

def parse_memory_usage(output: str) -> dict:
    """Parse memory usage from preprocessing output."""
    memory_info = {}
    lines = output.split('\n')
    
    for line in lines:
        if 'Memory Summary:' in line:
            idx = lines.index(line)
            # Parse the next few lines
            for i in range(1, 5):
                if idx + i < len(lines):
                    next_line = lines[idx + i].strip()
                    if 'Start:' in next_line:
                        memory_info['start'] = float(next_line.split()[2])
                    elif 'Peak:' in next_line:
                        memory_info['peak'] = float(next_line.split()[2])
                    elif 'Final:' in next_line:
                        memory_info['final'] = float(next_line.split()[2])
                    elif 'Peak increase:' in next_line:
                        memory_info['peak_increase'] = float(next_line.split()[3])
            break
    
    return memory_info

def parse_token_stats(output: str) -> dict:
    """Parse token statistics from preprocessing output."""
    import re
    
    stats = {}
    
    # Remove ANSI escape codes first
    clean_output = re.sub(r'\033\[[0-9;]*m', '', output)
    lines = clean_output.split('\n')
    
    for line in lines:
        line = line.strip()
        if 'Dropped' in line and 'tokens during filtering' in line:
            # Extract from "Dropped 1424 tokens during filtering"
            parts = line.split()
            if len(parts) >= 2 and parts[0] == 'Dropped':
                try:
                    stats['filter_dropped'] = int(parts[1])
                except ValueError:
                    pass
        elif 'Dropped' in line and ('final incomplete chunk' in line or 'carryover tokens' in line):
            # Extract from "Dropped 1009 tokens from final incomplete chunk"
            parts = line.split()
            if len(parts) >= 2 and parts[0] == 'Dropped':
                try:
                    stats['final_dropped'] = int(parts[1])
                except ValueError:
                    pass
        elif line.startswith('Original dataset rows:'):
            try:
                stats['original_rows'] = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith('Packed dataset rows:'):
            try:
                stats['packed_rows'] = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
        elif 'Total tokens in packed dataset:' in line:
            # Extract from "Total tokens in packed dataset: 0.000B"
            try:
                # Find the part after the colon
                colon_idx = line.find(':')
                if colon_idx >= 0:
                    token_part = line[colon_idx+1:].strip()
                    if token_part.endswith('B'):
                        stats['total_tokens_b'] = float(token_part[:-1])
            except (ValueError, IndexError):
                pass
    
    return stats

def main():
    """Main test function."""
    print("🧪 Testing preprocessing modes...")
    
    # Clean up any previous artifacts
    if os.path.exists('/home/runner/work/krill/krill/artifacts'):
        shutil.rmtree('/home/runner/work/krill/krill/artifacts')
    
    # Test results storage
    results = {}
    
    # Test both modes
    for mode_name, memory_efficient in [('standard', False), ('memory_efficient', True)]:
        print(f"\n🔍 Testing {mode_name} mode...")
        
        # Create test directories
        output_dir = f'/tmp/test_output_{mode_name}'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create config
        config_path = create_test_config(memory_efficient, output_dir)
        
        # Run preprocessing
        output, return_code = run_preprocessing(config_path)
        
        print(f"Return code: {return_code}")
        if return_code != 0:
            print(f"ERROR in {mode_name} mode:")
            print(output[-1000:])  # Last 1000 chars
            results[mode_name] = {'error': True, 'output': output}
            continue
        
        # Parse results
        memory_info = parse_memory_usage(output)
        token_stats = parse_token_stats(output)
        
        results[mode_name] = {
            'error': False,
            'memory': memory_info,
            'tokens': token_stats,
            'output': output
        }
        
        print(f"✅ {mode_name} mode completed")
        print(f"   Memory peak increase: {memory_info.get('peak_increase', 'N/A')} MB")
        print(f"   Filter dropped tokens: {token_stats.get('filter_dropped', 'N/A')}")
        print(f"   Final dropped tokens: {token_stats.get('final_dropped', 'N/A')}")
        print(f"   Packed rows: {token_stats.get('packed_rows', 'N/A')}")
    
    # Compare results
    print("\n📊 COMPARISON RESULTS:")
    print("=" * 50)
    
    if results.get('standard', {}).get('error') or results.get('memory_efficient', {}).get('error'):
        print("❌ One or both modes failed - cannot compare")
        if results.get('standard', {}).get('error'):
            print("Standard mode failed")
        if results.get('memory_efficient', {}).get('error'):
            print("Memory-efficient mode failed")
        return 1
    
    std_mem = results['standard']['memory'].get('peak_increase', 0)
    eff_mem = results['memory_efficient']['memory'].get('peak_increase', 0)
    
    print(f"Memory Usage:")
    print(f"  Standard mode: {std_mem:.1f} MB")
    print(f"  Memory-efficient mode: {eff_mem:.1f} MB")
    
    if eff_mem < std_mem:
        print("✅ Memory-efficient mode uses less memory")
    else:
        print("❌ Memory-efficient mode uses MORE memory!")
    
    # Compare token statistics
    std_tokens = results['standard']['tokens']
    eff_tokens = results['memory_efficient']['tokens']
    
    print(f"\nToken Statistics:")
    for key in ['filter_dropped', 'packed_rows', 'total_tokens_b']:
        std_val = std_tokens.get(key, 'N/A')
        eff_val = eff_tokens.get(key, 'N/A')
        print(f"  {key}: Standard={std_val}, Efficient={eff_val}")
        
        if std_val != 'N/A' and eff_val != 'N/A':
            if abs(std_val - eff_val) < 0.001:  # Allow small floating point differences
                print(f"    ✅ Values match")
            else:
                print(f"    ❌ Values differ!")
    
    # Final validation
    print(f"\n🔍 VALIDATION:")
    issues = []
    
    if eff_mem >= std_mem:
        issues.append("Memory-efficient mode doesn't save memory")
    
    if std_tokens.get('filter_dropped') != eff_tokens.get('filter_dropped'):
        issues.append("Filter dropped tokens differ")
    
    if std_tokens.get('packed_rows') != eff_tokens.get('packed_rows'):
        issues.append("Packed rows differ")
    
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    else:
        print("✅ All tests passed!")
        return 0

if __name__ == "__main__":
    # Set PATH to include uv
    os.environ['PATH'] = '/home/runner/.local/bin:' + os.environ.get('PATH', '')
    sys.exit(main())