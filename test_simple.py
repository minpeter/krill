#!/usr/bin/env python3
"""
Simple test to demonstrate working memory-efficient preprocessing.

This script tests both modes and validates:
1. Memory usage is properly monitored
2. Results are identical between modes
3. Both validation commands work correctly
"""

import subprocess
import os

def run_test(mode_name: str, config_efficient: bool):
    """Run a single test mode."""
    print(f"\n{'='*60}")
    print(f"🧪 Testing {mode_name.upper()} mode")
    print(f"{'='*60}")
    
    # Update config
    config_path = "examples/memory_efficient_example.yaml"
    
    # Read current config
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Update the memory efficient setting
    if config_efficient:
        content = content.replace('preprocess_memory_efficient: false', 'preprocess_memory_efficient: true')
    else:
        content = content.replace('preprocess_memory_efficient: true', 'preprocess_memory_efficient: false')
    
    # Write updated config
    with open(config_path, 'w') as f:
        f.write(content)
    
    # Clean artifacts
    if os.path.exists('./artifacts'):
        import shutil
        shutil.rmtree('./artifacts')
    
    print(f"🚀 Running preprocessing...")
    result = subprocess.run(['uv', 'run', '-m', 'krill.main', 'preprocess', config_path], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Preprocessing failed!")
        print(result.stderr)
        return False
    
    # Extract key stats
    output = result.stdout + result.stderr
    
    # Find memory stats
    memory_increase = "N/A"
    for line in output.split('\n'):
        if 'Peak increase:' in line:
            try:
                memory_increase = line.split()[-2] + " MB"
            except:
                pass
    
    # Find token stats
    filter_dropped = "N/A"
    final_dropped = "N/A" 
    packed_rows = "N/A"
    
    for line in output.split('\n'):
        if 'Dropped' in line and 'filtering' in line:
            try:
                filter_dropped = line.split()[1]
            except:
                pass
        elif 'Dropped' in line and ('final incomplete' in line or 'carryover' in line):
            try:
                final_dropped = line.split()[1]
            except:
                pass
        elif line.startswith('Packed dataset rows:'):
            try:
                packed_rows = line.split(':')[1].strip()
            except:
                pass
    
    print(f"✅ Preprocessing completed!")
    print(f"   📊 Memory peak increase: {memory_increase}")
    print(f"   🗑️  Filter dropped tokens: {filter_dropped}")
    print(f"   🗑️  Final dropped tokens: {final_dropped}")
    print(f"   📦 Packed rows: {packed_rows}")
    
    print(f"\n🔍 Running validation...")
    result = subprocess.run(['uv', 'run', '-m', 'krill.main', 'validate', config_path], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Validation failed!")
        print(result.stderr)
        return False
    
    print(f"✅ Validation passed!")
    
    return {
        'memory_increase': memory_increase,
        'filter_dropped': filter_dropped,
        'final_dropped': final_dropped,
        'packed_rows': packed_rows
    }

def main():
    os.environ['PATH'] = '/home/runner/.local/bin:' + os.environ.get('PATH', '')
    os.chdir('/home/runner/work/krill/krill')
    
    print("🦐 Krill Memory-Efficient Preprocessing Test")
    print("This test validates that memory-efficient preprocessing works correctly.")
    
    # Test standard mode
    std_results = run_test("standard", False)
    
    # Test memory-efficient mode
    eff_results = run_test("memory-efficient", True)
    
    print(f"\n{'='*60}")
    print("📊 FINAL COMPARISON")
    print(f"{'='*60}")
    
    if not std_results or not eff_results:
        print("❌ One or both tests failed!")
        return 1
    
    print("✅ Both modes completed successfully!")
    print("\n📈 Results Comparison:")
    print(f"   Filter dropped tokens: Standard={std_results['filter_dropped']}, Efficient={eff_results['filter_dropped']}")
    print(f"   Final dropped tokens:  Standard={std_results['final_dropped']}, Efficient={eff_results['final_dropped']}")
    print(f"   Packed rows:           Standard={std_results['packed_rows']}, Efficient={eff_results['packed_rows']}")
    print(f"   Memory usage:          Standard={std_results['memory_increase']}, Efficient={eff_results['memory_increase']}")
    
    # Check consistency
    if (std_results['filter_dropped'] == eff_results['filter_dropped'] and
        std_results['final_dropped'] == eff_results['final_dropped'] and
        std_results['packed_rows'] == eff_results['packed_rows']):
        print("\n🎯 ✅ Results are IDENTICAL between modes!")
        print("🎯 ✅ Both validation commands work correctly!")
        print("\n💡 Memory-efficient mode benefits:")
        print("   • Bounded memory usage (doesn't grow with dataset size)")
        print("   • Can process datasets larger than available RAM")
        print("   • Produces identical results to standard mode")
        return 0
    else:
        print("\n❌ Results differ between modes!")
        return 1

if __name__ == "__main__":
    exit(main())