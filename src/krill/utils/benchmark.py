"""
Benchmarking utilities for comparing preprocessing modes.
"""
import os
import time
import subprocess
import tempfile
import psutil
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from memory_profiler import memory_usage


@dataclass
class BenchmarkResult:
    """Results from a preprocessing benchmark run."""
    mode: str  # "standard" or "memory-efficient"
    total_time: float  # seconds
    peak_memory: float  # MB
    memory_trace: List[Tuple[float, float]]  # [(time, memory_mb), ...]
    final_metrics: Dict[str, int]  # extracted from output
    success: bool
    error_message: str = ""


class PreprocessBenchmark:
    """Benchmark preprocessing modes with memory and timing analysis."""
    
    def __init__(self, config_path: str, artifacts_cleanup: bool = True):
        self.config_path = config_path
        self.artifacts_cleanup = artifacts_cleanup
        self.results: List[BenchmarkResult] = []
    
    def run_benchmark(self, memory_efficient: bool = False, timeout: int = 600) -> BenchmarkResult:
        """Run a single benchmark and collect detailed metrics."""
        mode = "memory-efficient" if memory_efficient else "standard"
        print(f"🔬 Running {mode} mode benchmark...")
        
        # Clean up artifacts before run
        if self.artifacts_cleanup:
            self._cleanup_artifacts()
        
        # Prepare command
        cmd = ["python", "-m", "krill.main", "preprocess"]
        if memory_efficient:
            cmd.append("--memory-efficient")
        cmd.append(self.config_path)
        
        # Track memory and timing
        memory_trace = []
        start_time = time.time()
        success = False
        error_message = ""
        final_metrics = {}
        
        def run_process():
            return subprocess.run(
                cmd,
                cwd="/home/runner/work/krill/krill",
                capture_output=True,
                text=True,
                timeout=timeout
            )
        
        try:
            # Use memory_profiler to track memory usage over time
            def memory_monitor():
                nonlocal memory_trace, start_time
                process = psutil.Process()
                while True:
                    try:
                        current_time = time.time() - start_time
                        current_memory = process.memory_info().rss / 1024 / 1024  # MB
                        memory_trace.append((current_time, current_memory))
                        time.sleep(0.1)  # Sample every 100ms
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        break
            
            # Run the process and monitor memory
            result = run_process()
            end_time = time.time()
            
            if result.returncode == 0:
                success = True
                final_metrics = self._extract_metrics_from_output(result.stdout)
                print(f"✅ {mode} mode completed successfully")
            else:
                error_message = result.stderr
                print(f"❌ {mode} mode failed: {error_message}")
            
            # Get peak memory from the output itself (more accurate)
            import re
            memory_match = re.search(r"Peak: ([0-9.]+) MB", result.stdout)
            if memory_match:
                peak_memory = float(memory_match.group(1))
            else:
                # Fallback to memory profiler
                mem_usage = memory_usage(run_process, interval=0.1, timeout=timeout)
                peak_memory = max(mem_usage) if mem_usage else 0
            
        except subprocess.TimeoutExpired:
            error_message = f"Process timed out after {timeout} seconds"
            peak_memory = 0
            end_time = start_time + timeout
        except Exception as e:
            error_message = str(e)
            peak_memory = 0
            end_time = time.time()
        
        total_time = end_time - start_time
        
        benchmark_result = BenchmarkResult(
            mode=mode,
            total_time=total_time,
            peak_memory=peak_memory,
            memory_trace=memory_trace,
            final_metrics=final_metrics,
            success=success,
            error_message=error_message
        )
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def run_comparison(self) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """Run both standard and memory-efficient modes and compare."""
        print("🚀 Starting preprocessing benchmark comparison...")
        
        standard_result = self.run_benchmark(memory_efficient=False)
        memory_efficient_result = self.run_benchmark(memory_efficient=True)
        
        return standard_result, memory_efficient_result
    
    def generate_comparison_plots(self, output_dir: str = "./artifacts/benchmark"):
        """Generate comparison plots for memory usage and timing."""
        if len(self.results) < 2:
            print("❌ Need at least 2 benchmark results to generate comparison plots")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate memory usage comparison
        self._plot_memory_comparison(output_dir)
        
        # Generate timing and metrics comparison
        self._plot_metrics_comparison(output_dir)
        
        # Generate summary report
        self._generate_summary_report(output_dir)
        
        print(f"📊 Benchmark plots saved to {output_dir}")
    
    def _plot_memory_comparison(self, output_dir: str):
        """Plot memory usage over time for both modes."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        colors = {'standard': '#2E86AB', 'memory-efficient': '#A23B72'}
        
        # Plot 1: Memory usage over time
        for result in self.results:
            if result.memory_trace:
                times, memories = zip(*result.memory_trace)
                ax1.plot(times, memories, label=f"{result.mode} mode", 
                        color=colors.get(result.mode, '#333333'), linewidth=2)
                
                # Add peak memory annotation
                peak_idx = memories.index(max(memories))
                ax1.annotate(f'Peak: {max(memories):.1f} MB', 
                           xy=(times[peak_idx], max(memories)),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', fc=colors.get(result.mode, '#333333'), alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.set_title('Memory Usage Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Peak memory and timing comparison
        modes = []
        peak_memories = []
        total_times = []
        
        for result in self.results:
            if result.success:
                modes.append(result.mode)
                peak_memories.append(result.peak_memory)
                total_times.append(result.total_time)
        
        if modes:
            x_pos = range(len(modes))
            bars1 = ax2.bar([x - 0.2 for x in x_pos], peak_memories, 0.4, 
                           label='Peak Memory (MB)', color='#F18F01', alpha=0.8)
            
            # Secondary y-axis for time
            ax2_twin = ax2.twinx()
            bars2 = ax2_twin.bar([x + 0.2 for x in x_pos], total_times, 0.4, 
                                label='Total Time (s)', color='#C73E1D', alpha=0.8)
            
            ax2.set_xlabel('Processing Mode')
            ax2.set_ylabel('Peak Memory (MB)', color='#F18F01')
            ax2_twin.set_ylabel('Total Time (seconds)', color='#C73E1D')
            ax2.set_title('Performance Comparison')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(modes)
            
            # Add value labels on bars
            for bar, value in zip(bars1, peak_memories):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            
            for bar, value in zip(bars2, total_times):
                ax2_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                             f'{value:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'memory_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_comparison(self, output_dir: str):
        """Plot preprocessing metrics comparison."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        successful_results = [r for r in self.results if r.success and r.final_metrics]
        
        if len(successful_results) < 2:
            print("⚠️  Not enough successful results to compare metrics")
            return
        
        modes = [r.mode for r in successful_results]
        
        # Metrics to compare
        metrics = {
            'Original Rows': 'original_rows',
            'Packed Rows': 'packed_rows', 
            'Filter Dropped Tokens': 'filter_dropped_tokens',
            'Final Dropped Tokens': 'chunk_dropped_tokens'
        }
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for idx, (metric_name, metric_key) in enumerate(metrics.items()):
            ax = [ax1, ax2, ax3, ax4][idx]
            values = [r.final_metrics.get(metric_key, 0) for r in successful_results]
            
            bars = ax.bar(modes, values, color=colors[idx], alpha=0.8)
            ax.set_title(metric_name)
            ax.set_ylabel('Count')
            
            # Add value labels
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value * 0.01,
                           f'{value:,}', ha='center', va='bottom', fontweight='bold')
            
            # Check if values are identical (they should be!)
            if len(set(values)) <= 1:
                ax.text(0.5, 0.95, '✅ Identical Results', transform=ax.transAxes,
                       ha='center', va='top', fontweight='bold', color='green',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
            else:
                ax.text(0.5, 0.95, '⚠️ Different Results', transform=ax.transAxes,
                       ha='center', va='top', fontweight='bold', color='red',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self, output_dir: str):
        """Generate a text summary report."""
        report_path = os.path.join(output_dir, 'benchmark_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("🔬 Krill Preprocessing Benchmark Report\n")
            f.write("=" * 50 + "\n\n")
            
            for result in self.results:
                f.write(f"📊 {result.mode.upper()} MODE RESULTS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Success: {'✅ Yes' if result.success else '❌ No'}\n")
                f.write(f"Total Time: {result.total_time:.2f} seconds\n")
                f.write(f"Peak Memory: {result.peak_memory:.1f} MB\n")
                
                if result.final_metrics:
                    f.write("\nDataset Metrics:\n")
                    for key, value in result.final_metrics.items():
                        f.write(f"  {key}: {value:,}\n")
                
                if not result.success:
                    f.write(f"\nError: {result.error_message}\n")
                
                f.write("\n")
            
            # Comparison summary
            if len([r for r in self.results if r.success]) >= 2:
                successful = [r for r in self.results if r.success]
                f.write("📈 PERFORMANCE COMPARISON\n")
                f.write("-" * 30 + "\n")
                
                standard = next((r for r in successful if r.mode == "standard"), None)
                memory_eff = next((r for r in successful if r.mode == "memory-efficient"), None)
                
                if standard and memory_eff:
                    time_diff = memory_eff.total_time - standard.total_time
                    memory_diff = memory_eff.peak_memory - standard.peak_memory
                    
                    f.write(f"Time difference: {time_diff:+.2f} seconds ")
                    f.write(f"({'slower' if time_diff > 0 else 'faster'} memory-efficient mode)\n")
                    
                    f.write(f"Memory difference: {memory_diff:+.1f} MB ")
                    f.write(f"({'higher' if memory_diff > 0 else 'lower'} memory-efficient mode)\n")
                    
                    f.write(f"\nMemory efficiency: {(1 - memory_eff.peak_memory/standard.peak_memory)*100:.1f}% ")
                    f.write(f"{'reduction' if memory_diff < 0 else 'increase'}\n")
    
    def _extract_metrics_from_output(self, output_text: str) -> Dict[str, int]:
        """Extract key metrics from preprocessing output."""
        import re
        metrics = {}
        
        # Extract dropped tokens from filtering
        filter_match = re.search(r"Dropped (\d+) tokens.*during filtering", output_text)
        if filter_match:
            metrics["filter_dropped_tokens"] = int(filter_match.group(1))
        
        # Extract dropped tokens from final chunk
        chunk_match = re.search(r"Dropped (\d+) tokens.*from final incomplete chunk", output_text)
        if chunk_match:
            metrics["chunk_dropped_tokens"] = int(chunk_match.group(1))
        
        # Extract original dataset rows
        orig_match = re.search(r"Original dataset rows: (\d+)", output_text)
        if orig_match:
            metrics["original_rows"] = int(orig_match.group(1))
        
        # Extract packed dataset rows
        packed_match = re.search(r"Packed dataset rows: (\d+)", output_text)
        if packed_match:
            metrics["packed_rows"] = int(packed_match.group(1))
        
        return metrics
    
    def _cleanup_artifacts(self):
        """Clean up artifacts directory before benchmark run."""
        import shutil
        artifacts_path = "/home/runner/work/krill/krill/artifacts"
        if os.path.exists(artifacts_path):
            shutil.rmtree(artifacts_path)


def run_benchmark_comparison(config_path: str, output_dir: str = "./artifacts/benchmark") -> None:
    """Convenience function to run a complete benchmark comparison."""
    benchmark = PreprocessBenchmark(config_path)
    
    print("🚀 Starting comprehensive preprocessing benchmark...")
    standard_result, memory_efficient_result = benchmark.run_comparison()
    
    print("\n📊 Generating comparison visualizations...")
    benchmark.generate_comparison_plots(output_dir)
    
    print(f"\n✅ Benchmark complete! Results saved to {output_dir}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("🔬 BENCHMARK SUMMARY")
    print("="*60)
    
    for result in [standard_result, memory_efficient_result]:
        print(f"\n📊 {result.mode.upper()} MODE:")
        print(f"   Time: {result.total_time:.2f}s")
        print(f"   Peak Memory: {result.peak_memory:.1f} MB")
        print(f"   Success: {'✅' if result.success else '❌'}")
        
        if result.final_metrics:
            print(f"   Packed Rows: {result.final_metrics.get('packed_rows', 'N/A'):,}")
    
    if standard_result.success and memory_efficient_result.success:
        time_diff = memory_efficient_result.total_time - standard_result.total_time
        memory_diff = memory_efficient_result.peak_memory - standard_result.peak_memory
        
        print(f"\n🔄 COMPARISON:")
        print(f"   Time difference: {time_diff:+.2f}s")
        print(f"   Memory difference: {memory_diff:+.1f} MB")
        
        if memory_diff < 0:
            efficiency = (1 - memory_efficient_result.peak_memory/standard_result.peak_memory) * 100
            print(f"   Memory efficiency: {efficiency:.1f}% reduction 🎉")
        else:
            print(f"   Memory efficiency: {abs(memory_diff):.1f} MB increase")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "./artifacts/benchmark"
        run_benchmark_comparison(config_path, output_dir)
    else:
        print("Usage: python -m krill.utils.benchmark <config_path> [output_dir]")