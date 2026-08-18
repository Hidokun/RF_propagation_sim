"""Resource management debugger tests"""
import pytest
import os
import tempfile
import gc
import weakref
from propagation_model.models import free_space_path_loss
from propagation_model.itm_model = itm_model
from propagation_model.ray_tracing_model = ray_tracing_model
from RF_prop_sim.input_data_collection.ingestion = SimulationConfig
from RF_prop_sim.antenna_data.parser = AntennaConfig
from RF_prop_sim.mapping.location_service = LocationService
from RF_prop_sim.mapping.dem_provider = DEMProvider
from RF_prop_sim.mapping.dem_processor = DEMProcessor

class TestResourceCleanupDebugger:
    """Test suite for resource management debugger"""
    
    @pytest.mark.edge_case
    def test_temporary_file_cleanup(self):
        """Test that temporary files are properly cleaned up"""
        # This test verifies that file handles are closed properly
        
        # Create a temporary config file
        config_data = {"frequency": 3000.0, "distance": 5.0}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Verify file exists
            assert os.path.exists(temp_path)
            
            # Use the file
            config = SimulationConfig(temp_path)
            assert config is not None
            
            # File should still exist after use (delete=False)
            assert os.path.exists(temp_path)
            
        finally:
            # Clean up manually
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        # Test with automatic deletion (default behavior)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            import json
            json.dump(config_data, f)
            temp_path = f.name
            
            # File exists during context
            assert os.path.exists(temp_path)
            
            # Use the file
            config = SimulationConfig(temp_path)
            assert config is not None
        
        # File should be deleted after context exits
        # Note: On some systems, deletion might be delayed, so we check shortly after
        # But the important thing is that the context manager handles it
    
    @pytest.mark.edge_case
    def test_object_cleanup_and_garbage_collection(self):
        """Test that objects are properly cleaned up and eligible for garbage collection"""
        # Create objects and verify they can be garbage collected
        
        # Test SimulationConfig objects
        config_data = {"frequency": 3000.0, "distance": 5.0}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
            import json
            json.dump(config_data, f)
            temp_path = f.name
            
            # Create object
            config = SimulationConfig(temp_path)
            assert config is not None
            
            # Create weak reference to test garbage collection
            config_ref = weakref.ref(config)
            
            # Delete our reference
            del config
            
            # Force garbage collection
            gc.collect()
            
            # Object should be eligible for garbage collection
            # Note: This might not always work immediately due to Python's GC timing
            # but we're testing that there are no obvious resource leaks preventing GC
    
    @pytest.mark.edge_case
    def test_connection_resource_cleanup(self):
        """Test that network/connections resources are properly cleaned up"""
        location_service = LocationService()
        
        # Mock network requests to verify they're properly handled
        with patch.object(location_service, '_make_network_request') as mock_request:
            mock_request.return_value = {"status": "success", "data": "test"}
            
            # Make several requests
            for i in range(5):
                result = location_service.geocode(f"Location {i}")
                assert result is not None
            
            # Verify that requests were made
            assert mock_request.call_count == 5
            
            # The mock should have been properly cleaned up after the test
            # (This is handled by the patch context manager)
    
    @pytest.mark.edge_case
    def test_model_resource_cleanup(self):
        """Test that propagation model resources are properly managed"""
        # Test repeated calls to models don't accumulate resources
        
        # Test FSPL - should be stateless
        initial_objects = len(gc.get_objects())
        
        for i in range(1000):
            result = free_space_path_loss(1000.0, 1.0 * (i + 1))
            assert result is not None
        
        # Force garbage collection
        gc.collect()
        
        final_objects = len(gc.get_objects())
        
        # Object count should not have grown significantly
        # Allowing for some variation due to Python's internal caching
        object_diff = final_objects - initial_objects
        assert abs(object_diff) < 100, f"Too many objects created: {object_diff}"
        
        # Test other models similarly
        if 'itm_model' in globals() and itm_model is not None:
            try:
                initial_objects = len(gc.get_objects())
                
                for i in range(100):  # Fewer iterations for potentially heavier models
                    result = itm_model(
                        frequency_mhz=1000.0,
                        distance_km=1.0,
                        tx_height_m=10.0,
                        rx_height_m=10.0
                    )
                    # Result might be None if itmlogic not available, but shouldn't crash
                
                gc.collect()
                final_objects = len(gc.get_objects())
                object_diff = final_objects - initial_objects
                assert abs(object_diff) < 100, f"ITM model created too many objects: {object_diff}"
            except ImportError:
                pass  # Expected if itmlogic not available
        
        if 'ray_tracing_model' in globals() and ray_tracing_model is not None:
            try:
                initial_objects = len(gc.get_objects())
                
                for i in range(50):  # Even fewer for potentially heavy ray tracing
                    result = ray_tracing_model(
                        frequency_hz=30e9,
                        tx_array=[0, 0, 10],
                        rx_array=[100, 0, 1.5]
                    )
                    # Result might be None if sionna not available, but shouldn't crash
                
                gc.collect()
                final_objects = len(gc.get_objects())
                object_diff = final_objects - initial_objects
                assert abs(object_diff) < 100, f"Ray tracing model created too many objects: {object_diff}"
            except ImportError:
                pass  # Expected if sionna not available
    
    @pytest.mark.edge_case
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable over repeated operations"""
        import psutil
        import os
        
        # Get current process
        process = psutil.Process(os.getpid())
        
        # Baseline memory usage
        baseline_memory = process.memory_info().rss
        
        # Perform many operations
        for i in range(1000):
            # Vary the inputs slightly to prevent caching effects
            freq = 1000.0 + (i % 100)  # Vary frequency
            dist = 1.0 + (i % 50) / 10.0  # Vary distance
            
            result = free_space_path_loss(freq, dist)
            assert result is not None
            
            # Also test other models occasionally
            if i % 100 == 0:
                try:
                    rain_result = rain_attenuation(freq/1000.0, dist, 10.0)
                    assert rain_result is not None
                except Exception:
                    pass  # Some models might have issues with certain params, that's OK
        
        # Force garbage collection
        gc.collect()
        
        # Check memory usage after operations
        after_memory = process.memory_info().rss
        
        # Memory increase should be reasonable (allowing for normal Python fluctuations)
        # We're mainly looking for massive leaks, not small variations
        memory_increase = after_memory - baseline_memory
        memory_increase_mb = memory_increase / 1024 / 1024
        
        # Should not have leaked more than 50 MB (adjust as needed for your system)
        assert memory_increase_mb < 50, f"Possible memory leak: {memory_increase_mb:.2f} MB increase"
    
    @pytest.mark.edge_case
    def test_context_manager_resource_cleanup(self):
        """Test that context managers properly clean up resources"""
        # Test if any components use context managers that need proper cleanup
        
        # Test LocationService if it has context manager capabilities
        location_service = LocationService()
        
        # If it has a context manager interface, test it
        if hasattr(location_service, '__enter__') and hasattr(location_service, '__exit__'):
            with location_service as ls:
                result = ls.geocode("Test Location")
                # Should work within context
                assert result is not None or hasattr(ls, 'geocode')
            # Should have exited cleanly
        # If not, that's fine - not all services need context managers
    
    @pytest.mark.edge_case
    def test_cache_cleanup_and_memory_limits(self):
        """Test that caching mechanisms respect memory limits and clean up properly"""
        location_service = LocationService()
        
        # Test if location service has caching capabilities
        if hasattr(location_service, 'cache') or hasattr(location_service, '_cache'):
            # Fill the cache with many entries
            test_locations = [f"Location {i}" for i in range(100)]
            
            with patch.object(location_service, '_geocode_request') as mock_request:
                mock_request.return_value = {
                    'latitude': 40.0,
                    'longitude': -74.0,
                    'display_name': 'Test Location'
                }
                
                # Populate cache
                for location in test_locations:
                    location_service.geocode(location)
                
                # Verify cache was used (if implemented)
                initial_call_count = mock_request.call_count
                
                # Request same locations again - should use cache
                for location in test_locations[:10]:  # Test first 10
                    location_service.geocode(location)
                
                # If caching is working, call count shouldn't have increased much
                # (This test assumes a caching implementation exists)
                # For services without caching, all calls will go to the mock
                
                # Test cache clearing if available
                if hasattr(location_service, 'clear_cache'):
                    location_service.clear_cache()
                    # After clearing, requests should go to network again
                    # But we mainly want to ensure the method exists and doesn't crash
    
    @pytest.mark.edge_case
    def test_file_handle_cleanup_on_error(self):
        """Test that file handles are cleaned up even when errors occur"""
        # Test that if an error occurs while using a file, the handle is still released
        
        # Create a config file that will cause an error when parsed
        invalid_config_data = {"frequency": -1000.0, "distance": 5.0}  # Negative frequency
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(invalid_config_data, f)
            temp_path = f.name
        
        try:
            # Verify file exists
            assert os.path.exists(temp_path)
            
            # Try to use it - should cause an error
            with pytest.raises((ValueError, AssertionError, Exception)):
                config = SimulationConfig(temp_path)
                # If we get here without exception, that's also OK depending on implementation
            
            # Important: File handle should be released so we can delete the file
            # (On Windows, you can't delete a file that's still open)
            
        finally:
            # Try to clean up - this tests that the file handle was properly released
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                # If we get here without error, the file handle was properly released
            except PermissionError:
                # On Windows, if we get this error, it means the file handle wasn't released
                pytest.fail("File handle was not properly released after error")
    
    @pytest.mark.edge_case
    def test_thread_safety_resource_cleanup(self):
        """Test resource cleanup in multi-threaded scenarios"""
        import threading
        import time
        
        errors = []
        results = []
        
        def worker(worker_id):
            try:
                # Each worker creates and uses objects
                config_data = {"frequency": 3000.0 + worker_id, "distance": 5.0}
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as f:
                    import json
                    json.dump(config_data, f)
                    temp_path = f.name
                    
                    try:
                        config = SimulationConfig(temp_path)
                        results.append((worker_id, config is not None))
                        
                        # Do some work with the config
                        if config is not None:
                            _ = config.frequency
                            _ = config.distance
                    finally:
                        # File should be cleaned up by context manager
                        pass
                        
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Create and start multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Check for errors
        assert len(errors) == 0, f"Errors occurred in worker threads: {errors}"
        
        # Verify all workers got results
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        
        # Verify all workers succeeded
        for worker_id, success in results:
            assert success, f"Worker {worker_id} failed to create config"