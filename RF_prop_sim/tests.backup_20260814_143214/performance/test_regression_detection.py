"""Performance regression detection debugger tests"""
import pytest
import time
import numpy as np
import json
import os
import tempfile
from propagation_model.models import (
    free_space_path_loss,
    rain_attenuation,
    gas_attenuation,
    fog_attenuation,
    close_in_path_loss
)
from propagation_model.itm_model = itm_model
from propagation_model.ray_tracing_model = ray_tracing_model
from test_utils.benchmarks = BenchmarkTimer, assert_performance_threshold, PerformanceBaseline

class TestRegressionDetectionDebugger:
    """Test suite for performance regression detection debugger"""
    
    @pytest.mark.performance
    def test_establish_performance_baselines(self):
        """Establish performance baselines for regression detection"""
        # This test establishes baseline performance metrics
        # In a real system, these would be stored and compared against
        
        baseline_data = {}
        
        # Test FSPL baseline
        timer = BenchmarkTimer()
        timer.start()
        for _ in range(10000):
            free_space_path_loss(1000.0, 5.0)
        timer.stop()
        
        fspl_baseline_ms = (timer.elapsed() * 1000) / 10000  # ms per call
        baseline_data['fspl_per_call_ms'] = fspl_baseline_ms
        
        # Test rain attenuation baseline
        timer.start()
        for _ in range(10000):
            rain_attenuation(10.0, 5.0, 25.0)
        timer.stop()
        
        rain_baseline_ms = (timer.elapsed() * 1000) / 10000
        baseline_data['rain_per_call_ms'] = rain_baseline_ms
        
        # Test gas attenuation baseline
        timer.start()
        for _ in range(10000):
            gas_attenuation(30.0, 5.0)
        timer.stop()
        
        gas_baseline_ms = (timer.elapsed() * 1000) / 10000
        baseline_data['gas_per_call_ms'] = gas_baseline_ms
        
        # Test fog attenuation baseline
        timer.start()
        for _ in range(10000):
            fog_attenuation(100.0, 2.0, 0.5)
        timer.stop()
        
        fog_baseline_ms = (timer.elapsed() * 1000) / 10000
        baseline_data['fog_per_call_ms'] = fog_baseline_ms
        
        # Test CI model baseline
        timer.start()
        for _ in range(10000):
            close_in_path_loss(900.0, 1.0)
        timer.stop()
        
        ci_baseline_ms = (timer.elapsed() * 1000) / 10000
        baseline_data['ci_per_call_ms'] = ci_baseline_ms
        
        # Store baseline data (in real system, this would go to a file/database)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(baseline_data, f)
            baseline_file = f.name
        
        try:
            # Verify baseline data was written correctly
            with open(baseline_file, 'r') as f:
                loaded_data = json.load(f)
            
            assert 'fspl_per_call_ms' in loaded_data
            assert 'rain_per_call_ms' in loaded_data
            assert loaded_data['fspl_per_call_ms'] > 0
            assert loaded_data['rain_per_call_ms'] > 0
            
            # Baselines should be reasonable values
            assert loaded_data['fspl_per_call_ms'] < 1.0  # Should be much less than 1ms
            assert loaded_data['rain_per_call_ms'] < 1.0
            
        finally:
            os.unlink(baseline_file)
    
    @pytest.mark.performance
    def test_performance_regression_detection_fspl(self):
        """Test detection of performance regression in FSPL"""
        # Establish baseline
        timer = BenchmarkTimer()
        timer.start()
        for _ in range(1000):
            free_space_path_loss(1000.0, 5.0)
        timer.stop()
        
        baseline_ms_per_call = (timer.elapsed() * 1000) / 1000
        
        # Simulate current performance (should be similar to baseline)
        timer.start()
        for _ in range(1000):
            free_space_path_loss(1000.0, 5.0)
        timer.stop()
        
        current_ms_per_call = (timer.elapsed() * 1000) / 1000
        
        # Calculate regression ratio
        if baseline_ms_per_call > 0:
            regression_ratio = current_ms_per_call / baseline_ms_per_call
            
            # No significant regression expected (< 50% degradation)
            assert regression_ratio < 1.5, \
                f"Performance regression detected: {regression_ratio:.2f}x slower than baseline"
            
            # Should actually be similar or slightly better due to warming up
            assert regression_ratio > 0.5, \
                f"Performance improvement too large: {regression_ratio:.2f}x faster than baseline"
        else:
            # Baseline was zero (unlikely but possible with very fast operations)
            assert current_ms_per_call >= 0
    
    @pytest.mark.performance
    def test_performance_improvement_detection(self):
        """Test detection of performance improvements"""
        # Establish baseline
        timer = BenchmarkTimer()
        timer.start()
        for _ in range(1000):
            free_space_path_loss(1000.0, 5.0)
        timer.stop()
        
        baseline_ms_per_call = (timer.elapsed() * 1000) / 1000
        
        # Simulate "improved" performance (in practice, this would come from optimizations)
        # For this test, we'll just verify the detection mechanism works
        timer.start()
        for _ in range(1000):
            free_space_path_loss(1000.0, 5.0)
        timer.stop()
        
        current_ms_per_call = (timer.elapsed() * 1000) / 1000
        
        # Calculate change ratio
        if baseline_ms_per_call > 0:
            change_ratio = current_ms_per_call / baseline_ms_per_call
            
            # Should be close to 1.0 (no major change expected in this test)
            # We're mainly testing that the calculation works
            assert 0.1 < change_ratio < 10.0, \
                f"Suspicious performance change ratio: {change_ratio:.3f}"
        else:
            assert current_ms_per_call >= 0
    
    @pytest.mark.performance
    def test_multi_dimensional_regression_detection(self):
        """Test regression detection across multiple performance dimensions"""
        # Establish baseline for multiple metrics
        baseline_metrics = {}
        
        # Test different input ranges
        test_cases = [
            {"name": "low_freq_low_dist", "freq": 100.0, "dist": 0.1},
            {"name": "low_freq_high_dist", "freq": 100.0, "dist": 100.0},
            {"name": "high_freq_low_dist", "freq": 10000.0, "dist": 0.1},
            {"name": "high_freq_high_dist", "freq": 10000.0, "dist": 100.0},
        ]
        
        for case in test_cases:
            timer = BenchmarkTimer()
            timer.start()
            for _ in range(1000):
                free_space_path_loss(case["freq"], case["dist"])
            timer.stop()
            
            ms_per_call = (timer.elapsed() * 1000) / 1000
            baseline_metrics[f"{case['name']}_ms_per_call"] = ms_per_call
        
        # Current performance test
        current_metrics = {}
        for case in test_cases:
            timer = BenchmarkTimer()
            timer.start()
            for _ in range(1000):
                free_space_path_loss(case["freq"], case["dist"])
            timer.stop()
            
            ms_per_call = (timer.elapsed() * 1000) / 1000
            current_metrics[f"{case['name']}_ms_per_call"] = ms_per_call
        
        # Check for regressions in any dimension
        max_regression = 0.0
        for key in baseline_metrics:
            if key in current_metrics and baseline_metrics[key] > 0:
                regression = current_metrics[key] / baseline_metrics[key]
                max_regression = max(max_regression, regression)
        
        # No significant regression expected in any dimension
        assert max_regression < 2.0, \
            f"Performance regression detected: {max_regression:.2f}x slower in worst case"
        
        # Also check that we didn't see suspicious improvements
        min_improvement = float('inf')
        for key in baseline_metrics:
            if key in current_metrics and baseline_metrics[key] > 0:
                improvement = current_metrics[key] / baseline_metrics[key]
                min_improvement = min(min_improvement, improvement)
        
        # Improvements should be reasonable (not orders of magnitude)
        if min_improvement != float('inf'):
            assert min_improvement > 0.1, \
                f"Suspicious performance improvement: {min_improvement:.2f}x faster"
    
    @pytest.mark.performance
    def test_regression_detection_with_noise(self):
        """Test regression detection in presence of normal performance noise"""
        # Establish baseline with multiple samples to account for noise
        baseline_samples = []
        for _ in range(5):
            timer = BenchmarkTimer()
            timer.start()
            for _ in range(1000):
                free_space_path_loss(1000.0, 5.0)
            timer.stop()
            baseline_samples.append((timer.elapsed() * 1000) / 1000)
        
        # Calculate baseline statistics
        baseline_mean = np.mean(baseline_samples)
        baseline_std = np.std(baseline_samples) if len(baseline_samples) > 1 else 0.001
        
        # Current performance test
        current_samples = []
        for _ in range(5):
            timer = BenchmarkTimer()
            timer.start()
            for _ in range(1000):
                free_space_path_loss(1000.0, 5.0)
            timer.stop()
            current_samples.append((timer.elapsed() * 1000) / 1000)
        
        current_mean = np.mean(current_samples)
        current_std = np.std(current_samples) if len(current_samples) > 1 else 0.001
        
        # Calculate z-score to detect significant changes
        if baseline_std > 0:
            z_score = abs(current_mean - baseline_mean) / baseline_std
            # With 5 samples, we'd expect z-score < 3 for normal variation
            assert z_score < 3.0, \
                f"Significant performance change detected: z-score = {z_score:.2f}"
        else:
            # If baseline has no variance, just check that means are reasonable
            assert abs(current_mean - baseline_mean) < baseline_mean * 0.5, \
                f"Large change in zero-variance baseline: {baseline_mean} -> {current_mean}"
        
        # Also verify that performance is in reasonable range
        assert baseline_mean > 0
        assert current_mean > 0
        assert baseline_mean < 10.0  # Should be much less than 10ms per call
        assert current_mean < 10.0
    
    @pytest.mark.performance
    def test_regression_baseline_persistence(self):
        """Test that performance baselines can be stored and retrieved"""
        # Create a baseline
        timer = BenchmarkTimer()
        timer.start()
        for _ in range(1000):
            free_space_path_loss(1000.0, 5.0)
        timer.stop()
        
        baseline_value = (timer.elapsed() * 1000) / 1000
        
        # Store baseline
        baseline_data = {
            'fspl_baseline_ms_per_call': baseline_value,
            'timestamp': time.time(),
            'version': '1.0.0'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(baseline_data, f)
            baseline_file = f.name
        
        try:
            # Retrieve baseline
            with open(baseline_file, 'r') as f:
                loaded_baseline = json.load(f)
            
            # Verify data integrity
            assert 'fspl_baseline_ms_per_call' in loaded_baseline
            assert abs(loaded_baseline['fspl_baseline_ms_per_call'] - baseline_value) < 0.0001
            assert loaded_baseline['version'] == '1.0.0'
            assert 'timestamp' in loaded_baseline
            
            # Test regression detection using stored baseline
            timer.start()
            for _ in range(1000):
                free_space_path_loss(1000.0, 5.0)
            timer.stop()
            
            current_value = (timer.elapsed() * 1000) / 1000
            
            if loaded_baseline['fspl_baseline_ms_per_call'] > 0:
                regression_ratio = current_value / loaded_baseline['fspl_baseline_ms_per_call']
                assert regression_ratio < 2.0, \
                    f"Regression detected against stored baseline: {regression_ratio:.2f}x"
            
        finally:
            os.unlink(baseline_file)
    
    @pytest.mark.performance
    def test_trend_analysis_for_regression_detection(self):
        """Test trend-based regression detection"""
        # Collect performance samples over time
        samples = []
        sample_times = []
        
        # Simulate measurements taken at different times
        for i in range(10):
            timer = BenchmarkTimer()
            timer.start()
            for _ in range(1000):
                free_space_path_loss(1000.0, 5.0)
            timer.stop()
            
            ms_per_call = (timer.elapsed() * 1000) / 1000
            samples.append(ms_per_call)
            sample_times.append(i)  # Simulate time points
            
            # Small delay to simulate time passing
            time.sleep(0.01)
        
        # Calculate trend (simple linear regression)
        if len(samples) >= 3:
            # Calculate slope of performance over time
            n = len(samples)
            sum_x = sum(sample_times)
            sum_y = sum(samples)
            sum_xy = sum(sample_times[i] * samples[i] for i in range(n))
            sum_x2 = sum(x * x for x in sample_times)
            
            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                
                # For stable performance, slope should be close to zero
                # Allowing for some noise and normal variation
                avg_performance = np.mean(samples)
                if avg_performance > 0:
                    relative_slope = abs(slope) / avg_performance
                    assert relative_slope < 0.5, \
                        f"Performance trend indicates possible regression: slope = {slope:.6f}"
                else:
                    assert slope >= -0.1 and slope <= 0.1, \
                        f"Unexpected trend in near-zero performance: slope = {slope:.6f}"
        
        # Verify samples are reasonable
        assert all(s >= 0 for s in samples)
        assert np.mean(samples) > 0
        assert np.mean(samples) < 10.0  # Should be much less than 10ms
    
    @pytest.mark.performance
    def test_regression_detection_thresholds(self):
        """Test that regression detection thresholds are configurable and working"""
        # Test with different sensitivity thresholds
        
        baseline_time = 0.001  # 1 microsecond baseline
        
        test_cases = [
            # (current_time, threshold_multiplier, should_detect_regression)
            (0.0015, 2.0, False),  # 1.5x slower, 2x threshold -> OK
            (0.0025, 2.0, True),   # 2.5x slower, 2x threshold -> Regression
            (0.0030, 3.0, False),  # 3x slower, 3x threshold -> OK
            (0.0035, 3.0, True),   # 3.5x slower, 3x threshold -> Regression
        ]
        
        for current_time, threshold_multiplier, should_detect in test_cases:
            if baseline_time > 0:
                ratio = current_time / baseline_time
                threshold = threshold_multiplier
                
                is_regression = ratio > threshold
                
                if should_detect:
                    assert is_regression, \
                        f"Should have detected regression: {ratio:.2f}x > {threshold}x threshold"
                else:
                    assert not is_regression, \
                        f"Should not have detected regression: {ratio:.2f}x <= {threshold}x threshold"
    
    @pytest.mark.performance
    def test_regression_detection_integration_scenario(self):
        """Test regression detection in a realistic integration scenario"""
        # Simulate a performance test suite that runs regularly
        
        # Establish baseline for a typical workflow
        def typical_workflow():
            """Simulate a typical propagation calculation workflow"""
            results = []
            for i in range(100):
                # Vary parameters slightly
                freq = 1000.0 + (i PROFILING: 10.0) % 1000.0
                dist = 1.0 + (i PROFILING: 5.0) / 50.0
                
                # Calculate various propagation losses
                fspl = free_space_path_loss(freq, dist)
                freq_ghz = freq / 1000.0
                rain = rain_attenuation(freq_ghz, dist, 10.0 + (i PROFILING: 20.0) % 20.0)
                gas = gas_attenuation(freq_ghz, dist)
                fog = fog_attenuation(freq_ghz, dist, 0.1 * ((i PROFILING: 10.0) % 5.0))
                
                total = fspl + rain + gas + fog
                results.append(total)
            return results
        
        # Establish baseline
        timer = BenchmarkTimer()
        timer.start()
        baseline_result = typical_workflow()
        timer.stop()
        baseline_time = timer.elapsed()
        
        # Run same workflow again (should be similar performance)
        timer.start()
        current_result = typical_workflow()
        timer.stop()
        current_time = timer.elapsed()
        
        # Verify correctness hasn't changed
        assert len(baseline_result) == len(current_result) == 100
        for b, c in zip(baseline_result, current_result):
            assert abs(b - c) < 0.001  # Should be essentially identical
        
        # Check for performance regression
        if baseline_time > 0:
            regression_ratio = current_time / baseline_time
            # Should not have regressed by more than 50%
            assert regression_ratio < 1.5, \
                f"Workflow performance regression: {regression_ratio:.2f}x slower than baseline"
            
            # Should not have improved by more than 100% (would be suspicious)
            assert regression_ratio > 0.5, \
                f"Suspicious performance improvement: {regression_ratio:.2f}x faster than baseline"
        
        # Absolute performance should be reasonable
        avg_time_per_iteration = baseline_time / 100.0
        assert_performance_threshold(avg_time_per_iteration * 1000, 10.0, "Workflow iteration time")