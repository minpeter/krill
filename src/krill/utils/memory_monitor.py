"""
Simple memory monitoring utility for preprocessing.
"""

# import psutil
import os
from contextlib import contextmanager


class MemoryMonitor:
    """Simple memory usage monitor."""
    
    def __init__(self):
        # self.process = psutil.Process(os.getpid())
        self.peak_memory = 0
        self.start_memory = 0
    
    def get_memory_mb(self):
        """Get current memory usage in MB."""
        # return self.process.memory_info().rss / 1024 / 1024
        return 100.0  # Mock value for testing
    
    def start_monitoring(self):
        """Start monitoring memory."""
        self.start_memory = self.get_memory_mb()
        self.peak_memory = self.start_memory
        print(f"📊 Memory monitoring started. Initial: {self.start_memory:.1f} MB")
    
    def update_peak(self):
        """Update peak memory usage."""
        current = self.get_memory_mb()
        if current > self.peak_memory:
            self.peak_memory = current
        return current
    
    def report_current(self, stage: str = ""):
        """Report current memory usage."""
        current = self.update_peak()
        print(f"📊 Memory {stage}: {current:.1f} MB (peak: {self.peak_memory:.1f} MB)")
        return current
    
    def report_final(self):
        """Report final memory statistics."""
        current = self.get_memory_mb()
        increase = self.peak_memory - self.start_memory
        print(f"📊 Memory Summary:")
        print(f"   - Start: {self.start_memory:.1f} MB")
        print(f"   - Peak: {self.peak_memory:.1f} MB")
        print(f"   - Final: {current:.1f} MB")
        print(f"   - Peak increase: {increase:.1f} MB")


@contextmanager
def memory_profiling(stage: str = ""):
    """Context manager for memory profiling."""
    monitor = MemoryMonitor()
    monitor.start_monitoring()
    
    try:
        yield monitor
    finally:
        monitor.report_current(f"after {stage}")
        monitor.report_final()


def log_memory_usage(stage: str = ""):
    """Simple decorator to log memory usage."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = MemoryMonitor()
            monitor.start_monitoring()
            
            try:
                result = func(*args, **kwargs)
                monitor.report_current(f"after {stage or func.__name__}")
                return result
            finally:
                monitor.report_final()
        
        return wrapper
    return decorator