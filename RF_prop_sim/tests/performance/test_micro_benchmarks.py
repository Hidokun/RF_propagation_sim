"""Micro-benchmark performance debugger tests"""
import pytest
import time
import numpy as np
from propagation_model import (
    free_space_path_loss,
    rain_attenuation,
    gas_attenuation,
    fog_attenuation,
    close_in_path_loss,
)
from propagation_model.itm_model import itm_path_loss as itm_model
from propagation_model.ray_tracing_model import ray_tracing_path_loss as ray_tracing_model
from test_utils.benchmarks import (
    BenchmarkTimer,
    assert_performance_threshold,
    require_usable_sionna,
)

class TestMicroBenchmarksDebugger:
    """Test suite for micro-benchmark performance debugger"""
    
    @pytest.mark.performance
    def test_fspl_microbenchmark(self):
        """Benchmark FSPL function call performance"""
        timer = BenchmarkTimer()
        
        # Warm up
        for _ in range(100):
            free_space_path_loss(1000.0, 5.0)
        
        # Actual benchmark
        timer.start()
        iterations = 10000
        for _ in range(iterations):
            free_space_path_loss(1000.0, 5.0)
        timer.stop()
        
        # Calculate average time per call
        avg_time_ms = (timer.elapsed() * 1000) / iterations
        
        # Should be very fast - less than 0.1 ms per call on modern systems
        assert_performance_threshold(avg_time_ms, 0.1, "FSPL microbenchmark")
        
        # Also test with varying inputs
        timer.start()
        for i in range(iterations):
            freq = 100.0 + (i % 1000)  # Vary frequency
            dist = 1.0 + (i % 100) / 10.0  # Vary distance
            free_space_path_loss(freq, dist)
        timer.stop()
        
        avg_time_ms_var = (timer.elapsed() * 1000) / iterations
        assert_performance_threshold(avg_time_ms_var, 0.15, "FSPL microbenchmark with varying inputs")
    
    @pytest.mark.performance
    def test_rain_attenuation_microbenchmark(self):
        """Benchmark rain attenuation function call performance"""
        timer = BenchmarkTimer()
        
        # Warm up
        for _ in range(100):
            rain_attenuation(10.0, 5.0, 25.0)
        
        # Actual benchmark
        timer.start()
        iterations = 10000
        for _ in range(iterations):
            rain_attenuation(10.0, 5.0, 25.0)
        timer.stop()
        
        avg_time_ms = (timer.elapsed() * 1000) / iterations
        assert_performance_threshold(avg_time_ms, 0.15, "Rain attenuation microbenchmark")
    
    @pytest.mark.performance
    def test_gas_attenuation_microbenchmark(self):
        """Benchmark gas attenuation function call performance"""
        timer = BenchmarkTimer()
        
        # Warm up
        for _ in range(100):
            gas_attenuation(30.0, 5.0)
        
        # Actual benchmark
        timer.start()
        iterations = 10000
        for _ in range(iterations):
            gas_attenuation(30.0, 5.0)
        timer.stop()
        
        avg_time_ms = (timer.elapsed() * 1000) / iterations
        assert_performance_threshold(avg_time_ms, 0.15, "Gas attenuation microbenchmark")
    
    @pytest.mark.performance
    def test_fog_attenuation_microbenchmark(self):
        """Benchmark fog attenuation function call performance"""
        timer = BenchmarkTimer()
        
        # Warm up
        for _ in range(100):
            fog_attenuation(100.0, 2.0, 0.5)
        
        # Actual benchmark
        timer.start()
        iterations = 10000
        for _ in range(iterations):
            fog_attenuation(100.0, 2.0, 0.5)
        timer.stop()
        
        avg_time_ms = (timer.elapsed() * 1000) / iterations
        assert_performance_threshold(avg_time_ms, 0.15, "Fog attenuation microbenchmark")
    
    @pytest.mark.performance
    def test_ci_model_microbenchmark(self):
        """Benchmark CI model function call performance"""
        timer = BenchmarkTimer()
        
        # Warm up
        for _ in range(100):
            close_in_path_loss(900.0, 1.0)
        
        # Actual benchmark
        timer.start()
        iterations = 10000
        for _ in range(iterations):
            close_in_path_loss(900.0, 1.0)
        timer.stop()
        
        avg_time_ms = (timer.elapsed() * 1000) / iterations
        assert_performance_threshold(avg_time_ms, 0.15, "CI model microbenchmark")
    
    @pytest.mark.performance
    @pytest.mark.requires_itmlogic
    def test_itm_model_microbenchmark(self):
        """Benchmark ITM model function call performance"""
        # Skip if itmlogic not available
        pytest.importorskip("itmlogic")
        
        timer = BenchmarkTimer()
        
        # Warm up
        for _ in range(10):
            itm_model(1000.0, 5.0, 10.0, 10.0)
        
        # Actual benchmark - fewer iterations as ITM is more complex
        timer.start()
        iterations = 1000
        for _ in range(iterations):
            itm_model(1000.0, 5.0, 10.0, 10.0)
        timer.stop()
        
        avg_time_ms = (timer.elapsed() * 1000) / iterations
        # ITM is more complex, so allow higher threshold
        assert_performance_threshold(avg_time_ms, 5.0, "ITM model microbenchmark")
    
    @pytest.mark.performance
    @pytest.mark.requires_sionna
    def test_ray_tracing_microbenchmark(self):
        """Benchmark ray tracing function call performance"""
        require_usable_sionna()

        timer = BenchmarkTimer()

        # Warm up (signature: frequency_ghz, tx_pos, rx_pos)
        for _ in range(5):
            ray_tracing_model(30.0, [0, 0, 10], [100, 0, 1.5])

        # Actual benchmark - much fewer iterations as ray tracing is very complex
        timer.start()
        iterations = 50
        for _ in range(iterations):
            ray_tracing_model(30.0, [0, 0, 10], [100, 0, 1.5])
        timer.stop()
        
        avg_time_ms = (timer.elapsed() * 1000) / iterations
        # Ray tracing is much more complex, so allow much higher threshold
        assert_performance_threshold(avg_time_ms, 100.0, "Ray tracing microbenchmark")
    
    @pytest.mark.performance
    def test_model_comparison_microbenchmark(self):
        """Benchmark absolute per-call cost of each model.

        Note: sub-microsecond timings are at perf_counter resolution limits,
        so relative ordering between models is NOT asserted (0 < 0*3 is
        always False). Meaningful contract: every model stays under its
        per-call budget after warm-up.
        """
        timer = BenchmarkTimer()

        frequency_ghz = 10.0
        distance_km = 5.0
        frequency_mhz = frequency_ghz * 1000.0

        cases = [
            ("FSPL", lambda: free_space_path_loss(frequency_mhz, distance_km), 0.05),
            ("Rain", lambda: rain_attenuation(frequency_ghz, distance_km, 25.0), 0.10),
            ("Gas", lambda: gas_attenuation(frequency_ghz, distance_km), 0.10),
            ("Fog", lambda: fog_attenuation(frequency_ghz, distance_km, 0.5), 0.10),
            ("CI", lambda: close_in_path_loss(frequency_mhz, distance_km), 0.05),
        ]

        iterations = 10000
        results = {}
        for name, fn, budget_ms in cases:
            fn(); fn()  # warm-up
            timer.start()
            for _ in range(iterations):
                fn()
            timer.stop()
            avg_ms = (timer.elapsed() * 1000) / iterations
            results[name] = avg_ms
            assert avg_ms < budget_ms, \
                f"{name} averaged {avg_ms:.4f} ms/call, exceeds {budget_ms} ms budget"

        print("\nPer-call averages (ms):")
        for name, avg in results.items():
            print(f"  {name}: {avg:.5f}")
    
    @pytest.mark.performance
    def test_scalar_vs_vector_performance(self):
        """Performance comparison of scalar vs batch processing"""
        # This test checks if there are vectorized versions or if we should consider them
        
        timer = BenchmarkTimer()
        
        # Prepare test data
        n_points = 1000
        frequencies = np.linspace(100.0, 10000.0, n_points)  # 100 MHz to 10 GHz
        distances = np.linspace(1.0, 100.0, n_points)       # 1 km to 100 km
        
        # Test scalar approach (call function for each point)
        timer.start()
        results_scalar = []
        for i in range(n_points):
            result = free_space_path_loss(frequencies[i], distances[i])
            results_scalar.append(result)
        scalar_time = timer.elapsed()
        
        # Test if we can process in batches (if vectorized versions exist)
        # For now, we'll just note the scalar performance
        # In the future, if vectorized versions are added, this test can be extended
        
        avg_time_per_call = (scalar_time * 1000) / n_points
        assert_performance_threshold(avg_time_per_call, 0.2, "Scalar FSPL performance")
        
        # Verify results are reasonable
        assert len(results_scalar) == n_points
        assert all(r is not None and not np.isnan(r) and r >= 0 for r in results_scalar)
    
    @pytest.mark.performance
    def test_performance_consistency(self):
        """Test that FSPL performance is stable across runs.

        Uses enough iterations per sample that OS scheduling noise averages
        out, then tolerates one outlier sample (max <= 4x median).
        """
        timer = BenchmarkTimer()

        # Warm-up
        for _ in range(1000):
            free_space_path_loss(1000.0, 5.0)

        # Run benchmark multiple times
        times = []
        for run in range(7):
            timer.start()
            for _ in range(20000):
                free_space_path_loss(1000.0, 5.0)
            timer.stop()
            times.append(timer.elapsed())

        median_time = np.median(times)

        # Tolerate at most one scheduler-interrupted outlier sample
        assert max(times) <= 4.0 * median_time, \
            f"Performance too inconsistent: max {max(times):.4f}s vs median {median_time:.4f}s"

        # Absolute performance budget
        avg_time_per_call_ms = (median_time * 1000) / 20000
        assert_performance_threshold(avg_time_per_call_ms, 0.2, "Consistent FSPL performance")

