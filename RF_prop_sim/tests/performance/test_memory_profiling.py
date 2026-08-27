"""Memory profiling performance debugger tests"""
import pytest
import numpy as np
import gc
import os
import sys
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

class TestMemoryProfilingDebugger:
    """Test suite for memory profiling performance debugger"""
    
    @pytest.mark.performance
    def test_baseline_memory_usage(self):
        """Establish baseline memory usage for propagation models"""
        # Force garbage collection to get clean baseline
        gc.collect()
        
        # Measure baseline memory
        if hasattr(sys, 'getsizeof'):
            # Simple approach: measure object sizes
            pass
        
        # More sophisticated approach would use memory profiling tools
        # For now, we'll test that operations don't cause unreasonable memory growth
        
        initial_objects = len(gc.get_objects())
        
        # Perform many operations
        for i in range(10000):
            free_space_path_loss(1000.0 + i, 1.0 + i/100.0)
            if i % 100 == 0:  # Occasionally test other models
                rain_attenuation((1000.0 + i)/1000.0, 1.0 + i/100.0, 10.0)
                gas_attenuation((1000.0 + i)/1000.0, 1.0 + i/100.0)
        
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Object count should not have exploded
        object_increase = final_objects - initial_objects
        # Allow for some increase due to caching, interned strings, etc.
        assert object_increase < 1000, f"Excessive object creation: {object_increase} objects"
    
    @pytest.mark.performance
    def test_memory_growth_with_iterations(self):
        """Test that memory growth is linear or sub-linear with iterations"""
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Test with small number of iterations
        small_iterations = 1000
        for i in range(small_iterations):
            free_space_path_loss(1000.0, float(i))
        
        gc.collect()
        after_small = len(gc.get_objects())
        small_growth = after_small - initial_objects
        
        # Test with larger number of iterations
        large_iterations = 10000
        for i in range(large_iterations):
            free_space_path_loss(1000.0, float(i))
        
        gc.collect()
        after_large = len(gc.get_objects())
        large_growth = after_large - initial_objects
        
        # Growth should be roughly linear or better
        # If it were quadratic, large_growth would be much more than 10x small_growth
        if small_growth > 0:
            growth_ratio = large_growth / small_growth
            expected_ratio = large_iterations / small_iterations  # 10.0
            # Allow some variance but not explosive growth
            assert growth_ratio < expected_ratio * 3, \
                f"Memory growth appears super-linear: {growth_ratio:.2f} vs expected {expected_ratio:.2f}"
        
        # Absolute growth should be reasonable
        assert large_growth < 5000, f"Excessive memory growth: {large_growth} objects"
    
    @pytest.mark.performance
    def test_memory_reuse_and_caching(self):
        """Test that memory is reused appropriately and caching works correctly"""
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Repeatedly call the same function with the same parameters
        # Should not cause continuous memory growth if implemented efficiently
        for _ in range(5000):
            result = free_space_path_loss(1000.0, 5.0)
            assert result is not None
        
        gc.collect()
        after_repeated = len(gc.get_objects())
        repeated_growth = after_repeated - initial_objects
        
        # Should not have created thousands of objects for the same computation
        assert repeated_growth < 500, f"Excessive memory growth from repeated calls: {repeated_growth}"
        
        # Test with varying parameters that might benefit from caching
        gc.collect()
        initial_objects_var = len(gc.get_objects())
        
        # Use a pattern that repeats values
        for i in range(1000):
            freq = 1000.0 + (i % 100)  # Frequencies repeat every 100 iterations
            dist = 1.0 + (i % 50) / 10.0  # Distances repeat every 50 iterations
            result = free_space_path_loss(freq, dist)
            assert result is not None
        
        gc.collect()
        after_pattern = len(gc.get_objects())
        pattern_growth = after_pattern - initial_objects_var
        
        # Even with repeating patterns, growth should be controlled
        assert pattern_growth < 1000, f"Excessive memory growth with patterned inputs: {pattern_growth}"
    
    @pytest.mark.performance
    @pytest.mark.requires_itmlogic
    def test_itm_model_memory_usage(self):
        """Test memory usage characteristics of ITM model"""
        # Skip if itmlogic not available
        pytest.importorskip("itmlogic")
        
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # ITM model might use more memory due to complex computations
        for i in range(1000):
            result = itm_model(1000.0, 5.0 + i/100.0, 10.0, 10.0)
            # Result might be None if there are issues, but shouldn't crash
        
        gc.collect()
        after_itm = len(gc.get_objects())
        itm_growth = after_itm - initial_objects
        
        # ITM might use more memory, but should still be reasonable
        assert itm_growth < 2000, f"ITM model excessive memory growth: {itm_growth} objects"
        
        # Test repeated calls with same parameters
        gc.collect()
        initial_objects_same = len(gc.get_objects())
        
        for _ in range(1000):
            result = itm_model(1000.0, 5.0, 10.0, 10.0)
        
        gc.collect()
        after_itm_same = len(gc.get_objects())
        itm_same_growth = after_itm_same - initial_objects_same
        
        # Repeated calls with same params should not cause excessive growth
        assert itm_same_growth < 500, f"ITM model repeated calls excessive growth: {itm_same_growth}"
    
    @pytest.mark.performance
    @pytest.mark.requires_sionna
    def test_ray_tracing_model_memory_usage(self):
        """Test memory usage characteristics of ray tracing model"""
        require_usable_sionna()

        gc.collect()
        initial_objects = len(gc.get_objects())

        # Ray tracing model likely uses significant memory for scene data, etc.
        # But we still want to ensure it doesn't leak memory uncontrollably
        for i in range(100):  # Fewer iterations as ray tracing is memory intensive
            result = ray_tracing_model(30.0, [0, 0, 10], [100, 0, 1.5])
            # Result might be None if there are issues, but shouldn't crash

        gc.collect()
        after_rt = len(gc.get_objects())
        rt_growth = after_rt - initial_objects

        # Ray tracing might use more memory, but growth should be controlled
        assert rt_growth < 5000, f"Ray tracing model excessive memory growth: {rt_growth} objects"

        # Test repeated calls with same parameters
        gc.collect()
        initial_objects_same = len(gc.get_objects())

        for _ in range(100):  # Even fewer for repeated tests
            result = ray_tracing_model(30.0, [0, 0, 10], [100, 0, 1.5])
        
        gc.collect()
        after_rt_same = len(gc.get_objects())
        rt_same_growth = after_rt_same - initial_objects_same
        
        # Repeated calls should not cause unbounded growth
        assert rt_same_growth < 1000, f"Ray tracing model repeated calls excessive growth: {rt_same_growth}"
    
    @pytest.mark.performance
    def test_temporary_object_cleanup(self):
        """Test that temporary objects are cleaned up properly"""
        gc.collect()
        initial_objects = len(gc.get_objects())
        initial_collected = gc.collect()  # Force collection and count
        
        # Create many temporary objects through function calls
        for i in range(5000):
            # These calls may create temporary objects internally
            result1 = free_space_path_loss(1000.0, float(i))
            result2 = rain_attenuation(1.0, float(i/100.0), 10.0)
            result3 = gas_attenuation(1.0, float(i/100.0))
            
            # Explicitly delete references to help GC
            del result1, result2, result3
        
        # Force garbage collection
        collected = gc.collect()
        final_objects = len(gc.get_objects())
        
        # Object count should not have grown excessively
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Excessive object retention: {object_growth} objects"
        
        # Should have collected a reasonable number of objects
        # Note: This depends on Python's GC implementation and timing
    
    @pytest.mark.performance
    def test_large_scale_simulation_memory(self):
        """Test memory usage during large-scale simulation scenarios"""
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Simulate processing a large number of links/paths
        n_links = 5000
        
        link_results = []
        for i in range(n_links):
            # Simulate different links with varying characteristics
            freq_mhz = 1000.0 + (i % 10) * 100.0  # 1-2 GHz in steps
            distance_km = 1.0 + (i % 50) * 0.5   # 1-26 km in steps
            
            # Calculate propagation losses
            fspl = free_space_path_loss(freq_mhz, distance_km)
            freq_ghz = freq_mhz / 1000.0
            rain = rain_attenuation(freq_ghz, distance_km, 5.0 + (i % 20))  # Varying rain
            gas = gas_attenuation(freq_ghz, distance_km)
            fog = fog_attenuation(freq_ghz, distance_km, 0.1 * (i % 10))  # Varying fog
            
            total_loss = fspl + rain + gas + fog
            
            # Store minimal essential data (not the full objects)
            link_results.append({
                'link_id': i,
                'frequency_mhz': freq_mhz,
                'distance_km': distance_km,
                'total_loss_db': total_loss
            })
        
        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Object growth should be reasonable for the amount of work done
        object_growth = final_objects - initial_objects
        # Expect some growth due to storing results, but not excessive
        assert object_growth < 3000, f"Excessive object growth in large simulation: {object_growth}"
        
        # Verify we processed the expected number of links
        assert len(link_results) == n_links
        
        # Clear results and check memory can be freed
        del link_results
        gc.collect()
        after_clear_objects = len(gc.get_objects())
        
        # Object count should decrease after clearing large data structure
        # Note: This might not happen immediately due to Python's GC timing,
        # but we shouldn't see massive growth
        growth_after_clear = after_clear_objects - initial_objects
        assert growth_after_clear < 2000, f"Memory not properly released after clearing: {growth_after_clear}"
    
    @pytest.mark.performance
    def test_memory_efficiency_ratio(self):
        """Test memory efficiency - work done per unit of memory used"""
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Do a measurable amount of work
        work_units = 0
        for i in range(10000):
            free_space_path_loss(1000.0 + i, 1.0 + i/1000.0)
            work_units += 1
            
            # Occasionally do more complex work
            if i % 100 == 0:
                rain_attenuation((1000.0 + i)/1000.0, 1.0 + i/1000.0, 10.0)
                work_units += 1
        
        gc.collect()
        final_objects = len(gc.get_objects())
        object_increase = final_objects - initial_objects
        
        # Calculate work done per object created
        if object_increase > 0:
            work_per_object = work_units / object_increase
            # Should be able to do significant work per object allocated
            assert work_per_object > 10, f"Low memory efficiency: {work_per_object:.1f} work units per object"
        else:
            # No object increase is actually good - means we're reusing memory efficiently
            assert work_units > 0, "Should have done some work"
        
        # Absolute object increase should be reasonable
        assert object_increase < 2000, f"Too many objects created for work done: {object_increase}"

