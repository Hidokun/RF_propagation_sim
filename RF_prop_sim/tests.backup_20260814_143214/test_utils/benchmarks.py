"""Benchmarking utilities"""
import time
import numpy as np
from functools import wraps

def benchmark_function(func):
    """Decorator to benchmark function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper

def benchmark_memory(func):
    """Decorator to benchmark memory usage (simplified)"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Simple memory approximation - in practice would use memory_profiler
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        # Return result with execution time and estimated memory (placeholder)
        return result, execution_time, 0  # 0 MB placeholder
    return wrapper

class PerformanceTimer:
    """Context manager for timing code blocks"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time

class BenchmarkTimer:
    """Simple timer for benchmarking code execution"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer"""
        self.start_time = time.perf_counter()
    
    def stop(self):
        """Stop the timer"""
        self.end_time = time.perf_counter()
    
    def elapsed(self):
        """Get elapsed time in seconds"""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

def assert_performance_threshold(value, threshold, test_name=""):
    """Assert that a value is within a performance threshold"""
    assert value <= threshold, f"{test_name} value {value} exceeds threshold {threshold}"

def generate_test_report(test_results, suite_name="Test Suite"):
    """Generate a simple test report"""
    report = f"{suite_name} Report\n"
    report += "=" * len(f"{suite_name} Report") + "\n\n"
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result.get('passed', False))
    failed_tests = total_tests - passed_tests
    
    report += f"Total Tests: {total_tests}\n"
    report += f"Passed: {passed_tests}\n"
    report += f"Failed: {failed_tests}\n"
    report += f"Success Rate: {passed_tests/total_tests*100:.1f}%\n\n"
    
    if failed_tests > 0:
        report += "Failed Tests:\n"
        for result in test_results:
            if not result.get('passed', False):
                report += f"  - {result.get('test_name', 'Unknown')}: {result.get('error', 'Unknown error')}\n"
    
    return report