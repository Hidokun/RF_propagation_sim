"""Macro-benchmark performance debugger tests"""
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

class TestMacroBenchmarksDebugger:
    """Test suite for macro-benchmark performance debugger"""
    
    @pytest.mark.performance
    def test_end_to_end_simulation_baseline(self):
        """Benchmark end-to-end simulation workflow"""
        timer = BenchmarkTimer()
        
        # Warm up
        for _ in range(10):
            # Simulate a basic end-to-end workflow
            freq_mhz = 3000.0
            distance_km = 5.0
            
            # 1. Free space path loss
            fspl = free_space_path_loss(freq_mhz, distance_km)
            
            # 2. Rain attenuation (moderate rain)
            freq_ghz = freq_mhz / 1000.0
            rain = rain_attenuation(freq_ghz, distance_km, 25.0)
            
            # 3. Gas attenuation
            gas = gas_attenuation(freq_ghz, distance_km)
            
            # 4. Fog attenuation (light fog)
            fog = fog_attenuation(freq_ghz, distance_km, 0.1)
            
            # 5. CI model
            ci = close_in_path_loss(freq_mhz, distance_km)
            
            # 6. Total loss calculation
            total_loss = fspl + rain + gas + fog + ci
            
            # 7. Link budget calculation (simplified)
            tx_power = 30.0  # dBm
            tx_gain = 20.0   # dBi
            rx_gain = 20.0   # dBi
            received_power = tx_power + tx_gain - total_loss + rx_gain
        
        # Actual benchmark
        timer.start()
        iterations = 1000
        for _ in range(iterations):
            # Simulate a basic end-to-end workflow
            freq_mhz = 3000.0 + (_ % 1000)  # Vary frequency slightly
            distance_km = 1.0 + (_ % 100) / 10.0  # Vary distance slightly
            
            # 1. Free space path loss
            fspl = free_space_path_loss(freq_mhz, distance_km)
            
            # 2. Rain attenuation
            freq_ghz = freq_mhz / 1000.0
            rain_rate = 5.0 + (_ % 50)  # Vary rain rate
            rain = rain_attenuation(freq_ghz, distance_km, rain_rate)
            
            # 3. Gas attenuation
            gas = gas_attenuation(freq_ghz, distance_km)
            
            # 4. Fog attenuation
            fog_density = 0.1 + (_ % 20) / 100.0  # Vary fog density
            fog = fog_attenuation(freq_ghz, distance_km, fog_density)
            
            # 5. CI model
            ci = close_in_path_loss(freq_mhz, distance_km)
            
            # 6. Total loss calculation
            total_loss = fspl + rain + gas + fog + ci
            
            # 7. Link budget calculation (simplified)
            tx_power = 30.0  # dBm
            tx_gain = 20.0   # dBi
            rx_gain = 20.0   # dBi
            received_power = tx_power + tx_gain - total_loss + rx_gain
        timer.stop()
        
        # Calculate average time per simulation
        avg_time_ms = (timer.elapsed() * 1000) / iterations
        
        # Should be reasonably fast for end-to-end simulation
        assert_performance_threshold(avg_time_ms, 10.0, "End-to-end simulation baseline")
    
    @pytest.mark.performance
    @pytest.mark.requires_itmlogic
    def test_itm_end_to_end_simulation(self):
        """Benchmark end-to-end simulation with ITM model"""
        # Skip if itmlogic not available
        pytest.importorskip("itmlogic")
        
        timer = BenchmarkTimer()
        
        # Warm up
        for _ in range(5):
            freq_mhz = 1000.0
            distance_km = 5.0
            
            # Basic models
            fspl = free_space_path_loss(freq_mhz, distance_km)
            freq_ghz = freq_mhz / 1000.0
            rain = rain_attenuation(freq_ghz, distance_km, 10.0)
            gas = gas_attenuation(freq_ghz, distance_km)
            fog = fog_attenuation(freq_ghz, distance_km, 0.05)
            
            # ITM model
            itm_loss = itm_model(freq_mhz, distance_km, 10.0, 10.0)
        
        # Actual benchmark
        timer.start()
        iterations = 500  # Fewer iterations as ITM is more complex
        for _ in range(iterations):
            freq_mhz = 500.0 + (_ % 1000)  # Vary frequency
            distance_km = 1.0 + (_ % 50) / 10.0  # Vary distance
            
            # Basic models
            fspl = free_space_path_loss(freq_mhz, distance_km)
            freq_ghz = freq_mhz / 1000.0
            rain_rate = 5.0 + (_ % 20)  # Vary rain rate
            rain = rain_attenuation(freq_ghz, distance_km, rain_rate)
            gas = gas_attenuation(freq_ghz, distance_km)
            fog_density = 0.05 + (_ % 15) / 100.0  # Vary fog density
            fog = fog_attenuation(freq_ghz, distance_km, fog_density)
            
            # ITM model
            itm_loss = itm_model(freq_mhz, distance_km, 10.0 + (_ % 20), 10.0 + (_ % 20))
            
            # Total loss calculation
            total_loss = fspl + rain + gas + fog + (itm_loss if itm_loss is not None else 0)
        
        timer.stop()
        
        # Calculate average time per simulation
        avg_time_ms = (timer.elapsed() * 1000) / iterations
        
        # ITM-enabled simulation will be slower
        assert_performance_threshold(avg_time_ms, 50.0, "ITM end-to-end simulation")
    
    @pytest.mark.performance
    @pytest.mark.requires_sionna
    def test_ray_tracing_end_to_end_simulation(self):
        """Benchmark end-to-end simulation with ray tracing model"""
        require_usable_sionna()

        timer = BenchmarkTimer()

        def one_iteration(i, freq_hz):
            tx_array = [0, 0, 10]
            rx_array = [50 + (i % 50), 0, 1.5]

            # Basic models
            distance_m = np.sqrt(sum((rx_array[k] - tx_array[k])**2 for k in range(3)))
            distance_km = max(0.001, distance_m / 1000.0)
            freq_mhz = freq_hz / 1e6

            free_space_path_loss(freq_mhz, distance_km)
            freq_ghz = freq_mhz / 1000.0
            rain_attenuation(freq_ghz, distance_km, 5.0)
            gas_attenuation(freq_ghz, distance_km)
            fog_attenuation(freq_ghz, distance_km, 0.1)

            # Ray tracing model (frequency in GHz)
            return ray_tracing_model(freq_hz / 1e9, tx_array, rx_array)

        # Warm up
        for i in range(3):
            one_iteration(i, 30e9)

        # Actual benchmark
        timer.start()
        iterations = 50  # Much fewer iterations as ray tracing is very complex
        for i in range(iterations):
            freq_hz = 20e9 + (i % 20) * 1e9  # 20-40 GHz
            one_iteration(i, freq_hz)
        timer.stop()
        
        # Calculate average time per simulation
        avg_time_ms = (timer.elapsed() * 1000) / iterations
        
        # Ray tracing-enabled simulation will be much slower
        assert_performance_threshold(avg_time_ms, 500.0, "Ray tracing end-to-end simulation")
    
    @pytest.mark.performance
    def test_scaling_analysis(self):
        """Analyze how performance scales with input parameters.

        FSPL cost must not depend on input values; best-of-3 sampling per
        regime keeps the comparison stable under suite-time system load.
        """
        timer = BenchmarkTimer()

        test_configs = [
            {"name": "Low freq, short dist", "freq": 100.0, "dist": 0.1},
            {"name": "Low freq, long dist", "freq": 100.0, "dist": 100.0},
            {"name": "High freq, short dist", "freq": 10000.0, "dist": 0.1},
            {"name": "High freq, long dist", "freq": 10000.0, "dist": 100.0},
        ]

        times = []
        for config in test_configs:
            free_space_path_loss(config["freq"], config["dist"])  # warm-up
            best = np.inf
            for _ in range(3):
                timer.start()
                for _ in range(10000):
                    free_space_path_loss(config["freq"], config["dist"])
                timer.stop()
                best = min(best, timer.elapsed())
            times.append((config["name"], best))

        # All configurations should have similar performance
        # (FSPL calculation time should not depend significantly on input values)
        elapsed_times = [t[1] for t in times]
        max_time = max(elapsed_times)
        min_time = min(elapsed_times)

        if min_time > 0:
            ratio = max_time / min_time
            assert ratio < 5.0, f"Performance scaling too uneven: {ratio:.2f}x difference"
        
        # Log results for information
        print("\nFSPL Performance Scaling Analysis:")
        for name, elapsed in times:
            avg_ms = (elapsed * 1000) / 1000
            print(f"  {name}: {avg_ms:.3f} ms per call")
    
    @pytest.mark.performance
    def test_memory_access_pattern_performance(self):
        """Test performance with different memory access patterns"""
        timer = BenchmarkTimer()
        
        # Test sequential access (should be fastest due to caching)
        timer.start()
        for i in range(1000):
            freq = 1000.0 + i  # Sequential
            dist = 1.0 + i / 100.0  # Sequential
            free_space_path_loss(freq, dist)
        sequential_time = timer.elapsed()
        
        # Test random access
        import random
        random.seed(42)  # For reproducibility
        timer.start()
        for _ in range(1000):
            freq = random.uniform(100.0, 10000.0)  # Random
            dist = random.uniform(0.1, 100.0)      # Random
            free_space_path_loss(freq, dist)
        random_time = timer.elapsed()
        
        # Test strided access
        timer.start()
        for i in range(0, 10000, 10):  # Every 10th element
            freq = 1000.0 + i
            dist = 1.0 + i / 1000.0
            free_space_path_loss(freq, dist)
        strided_time = timer.elapsed()
        
        # Sequential access should be reasonably fast
        avg_sequential = (sequential_time * 1000) / 1000
        assert_performance_threshold(avg_sequential, 0.2, "Sequential access performance")
        
        # Random access might be slower due to cache misses, but shouldn't be dramatically slower
        if sequential_time > 0:
            random_ratio = random_time / sequential_time
            assert random_ratio < 10.0, f"Random access excessively slow: {random_ratio:.2f}x sequential"
        
        # Log results for information
        print("\nMemory Access Pattern Performance:")
        print(f"  Sequential: {(sequential_time*1000)/1000:.3f} ms per call")
        print(f"  Random:     {(random_time*1000)/1000:.3f} ms per call")
        print(f"  Strided:    {(strided_time*1000)/1000:.3f} ms per call")
    
    @pytest.mark.performance
    def test_concurrent_execution_simulation(self):
        """Simulate concurrent execution performance characteristics"""
        import threading
        import time
        
        results = []
        errors = []
        
        def worker(worker_id, iterations):
            """Worker function to simulate concurrent load"""
            try:
                worker_results = []
                for i in range(iterations):
                    # Each worker does slightly different work
                    freq = 1000.0 + worker_id * 100 + i
                    dist = 1.0 + (worker_id + i) / 50.0
                    result = free_space_path_loss(freq, dist)
                    worker_results.append(result)
                results.append((worker_id, len(worker_results), None))
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Test with multiple threads
        num_threads = 4
        iterations_per_thread = 250
        
        timer = BenchmarkTimer()
        timer.start()
        
        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i, iterations_per_thread))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        timer.stop()
        
        # Check for errors
        assert len(errors) == 0, f"Errors in concurrent execution: {errors}"
        
        # Verify all workers completed their work
        assert len(results) == num_threads, f"Expected {num_threads} results, got {len(results)}"
        
        total_work_items = sum(r[1] for r in results)
        expected_work_items = num_threads * iterations_per_thread
        assert total_work_items == expected_work_items, \
            f"Expected {expected_work_items} work items, got {total_work_items}"
        
        # Calculate effective throughput
        total_time_sec = timer.elapsed()
        work_items_per_sec = total_work_items / total_time_sec if total_time_sec > 0 else 0
        
        # Should be able to handle reasonable concurrent load
        # This is more of a smoke test - actual requirements would depend on deployment
        assert work_items_per_sec > 0, "No work processed in concurrent test"
        
        # Log results for information
        print(f"\nConcurrent Execution Simulation:")
        print(f"  Threads: {num_threads}")
        print(f"  Iterations per thread: {iterations_per_thread}")
        print(f"  Total work items: {total_work_items}")
        print(f"  Total time: {total_time_sec:.3f} sec")
        print(f"  Throughput: {work_items_per_sec:.0f} work items/sec")

