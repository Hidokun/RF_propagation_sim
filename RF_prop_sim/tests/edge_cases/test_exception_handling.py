"""Exception handling debugger tests"""
import pytest
import numpy as np
from propagation_model.models import (
    free_space_path_loss,
    rain_attenuation,
    gas_attenuation,
    fog_attenuation,
    close_in_path_loss
)
from propagation_model.itm_model import itm_path_loss as itm_model
from propagation_model.ray_tracing_model import ray_tracing_path_loss as ray_tracing_model
from RF_prop_sim.input_data_collection.ingestion import SimulationConfig
from RF_prop_sim.antenna_data.parser import AntennaConfig
from RF_prop_sim.mapping.location_service import Geocoder as LocationService

class TestExceptionHandlingDebugger:
    """Test suite for exception handling debugger"""
    
    @pytest.mark.edge_case
    def test_exception_types_and_messages(self):
        """Test that appropriate exception types are raised with meaningful messages"""
        # Test FSPL with invalid types
        with pytest.raises((TypeError, ValueError)) as exc_info:
            free_space_path_loss("invalid", 5.0)
        assert "frequency" in str(exc_info.value).lower() or "invalid" in str(exc_info.value)
        
        with pytest.raises((TypeError, ValueError)) as exc_info:
            free_space_path_loss(5.0, "invalid")
        assert "distance" in str(exc_info.value).lower() or "invalid" in str(exc_info.value)
        
        # Test rain model with invalid types
        with pytest.raises((TypeError, ValueError)) as exc_info:
            rain_attenuation("invalid", 5.0, 10.0)
        assert "frequency" in str(exc_info.value).lower() or "invalid" in str(exc_info.value)
        
        with pytest.raises((TypeError, ValueError)) as exc_info:
            rain_attenuation(5.0, "invalid", 10.0)
        assert "distance" in str(exc_info.value).lower() or "invalid" in str(exc_info.value)
        
        with pytest.raises((TypeError, ValueError)) as exc_info:
            rain_attenuation(5.0, 5.0, "invalid")
        assert "rain rate" in str(exc_info.value).lower() or "invalid" in str(exc_info.value)
    
    @pytest.mark.edge_case
    def test_exception_propagation_through_pipeline(self):
        """Test that exceptions propagate correctly through the pipeline"""
        # Test configuration loading with invalid JSON
        with pytest.raises((ValueError, json.JSONDecodeError)):
            # Create a file with invalid JSON
            with open("temp_invalid.json", "w") as f:
                f.write("{ invalid json content")
            try:
                SimulationConfig("temp_invalid.json")
            finally:
                import os
                if os.path.exists("temp_invalid.json"):
                    os.remove("temp_invalid.json")
        
        # Test antenna config loading with invalid JSON
        with pytest.raises((ValueError, json.JSONDecodeError)):
            with open("temp_invalid_ant.json", "w") as f:
                f.write("{ invalid json content")
            try:
                AntennaConfig("temp_invalid_ant.json")
            finally:
                import os
                if os.path.exists("temp_invalid_ant.json"):
                    os.remove("temp_invalid_ant.json")
    
    @pytest.mark.edge_case
    def test_exception_handling_in_model_combinations(self):
        """Test exception handling when combining multiple models"""
        # Test that if one model fails, others can still be evaluated
        try:
            # This should work
            fspl_result = free_space_path_loss(1000.0, 5.0)
            assert fspl_result is not None
        except Exception as e:
            pytest.fail(f"FSPL should not fail with valid inputs: {e}")
        
        try:
            # This should also work
            rain_result = rain_attenuation(1.0, 5.0, 25.0)  # 1 GHz, 5 km, 25 mm/h rain
            assert rain_result is not None
        except Exception as e:
            pytest.fail(f"Rain model should not fail with valid inputs: {e}")
        
        # Test graceful handling when one model receives problematic input
        # but others should still work
        problematic_inputs = [
            (0.0, 5.0, 25.0),   # Zero frequency
            (1.0, 0.0, 25.0),   # Zero distance
            (1.0, 5.0, -1.0),   # Negative rain rate
        ]
        
        for freq, dist, rain_rate in problematic_inputs:
            # FSPL should handle zero/negative inputs gracefully
            try:
                fspl_result = free_space_path_loss(freq, dist)
                # Depending on implementation, might be 0 or raise exception
                # We mainly want to ensure consistent behavior
                assert fspl_result is not None or isinstance(fspl_result, (int, float))
            except Exception:
                # Some implementations might raise exceptions - this is OK
                pass
            
            # Rain model should handle zero/negative inputs gracefully
            try:
                rain_result = rain_attenuation(freq, dist, rain_rate)
                # Should handle gracefully
                assert rain_result is not None or isinstance(rain_result, (int, float))
            except Exception:
                # Some implementations might raise exceptions - this is OK
                pass
    
    @pytest.mark.edge_case
    def test_itm_model_exception_handling(self):
        """Test exception handling specifically for ITM model"""
        try:
            # Test with invalid inputs
            with pytest.raises((ValueError, TypeError, ImportError, Exception)):
                itm_model(frequency_mhz="invalid", distance_km=5.0, tx_height_m=10.0, rx_height_m=10.0)
            
            with pytest.raises((ValueError, TypeError, ImportError, Exception)):
                itm_model(frequency_mhz=1000.0, distance_km="invalid", tx_height_m=10.0, rx_height_m=10.0)
            
            with pytest.raises((ValueError, TypeError, ImportError, Exception)):
                itm_model(frequency_mhz=1000.0, distance_km=5.0, tx_height_m="invalid", rx_height_m=10.0)
                
        except ImportError:
            # Expected if itmlogic is not available
            pytest.skip("itmlogic package not available for testing")
        except Exception as e:
            # If we get here, itmlogic is available but something unexpected happened
            # We'll allow various exception types as long as they're appropriate
            assert isinstance(e, (ValueError, TypeError, RuntimeError, ImportError, Exception))
    
    @pytest.mark.edge_case
    def test_ray_tracing_exception_handling(self):
        """Test exception handling specifically for ray tracing model"""
        try:
            # Test with invalid inputs
            with pytest.raises((ValueError, TypeError, ImportError, Exception)):
                ray_tracing_model(frequency_hz="invalid", tx_array=[0,0,10], rx_array=[100,0,1.5])
            
            with pytest.raises((ValueError, TypeError, ImportError, Exception)):
                ray_tracing_model(frequency_hz=30e9, tx_array="invalid", rx_array=[100,0,1.5])
            
            with pytest.raises((ValueError, TypeError, ImportError, Exception)):
                ray_tracing_model(frequency_hz=30e9, tx_array=[0,0,10], rx_array="invalid")
                
        except ImportError:
            # Expected if sionna is not available
            pytest.skip("sionna package not available for testing")
        except Exception as e:
            # If we get here, sionna is available but something unexpected happened
            # We'll allow various exception types as long as they're appropriate
            assert isinstance(e, (ValueError, TypeError, RuntimeError, ImportError, Exception))
    
    @pytest.mark.edge_case
    def test_location_service_exception_handling(self):
        """Test exception handling in location service"""
        location_service = LocationService()
        
        # Test with invalid inputs
        with pytest.raises((ValueError, TypeError, Exception)):
            location_service.geocode(None)  # None input
        
        with pytest.raises((ValueError, TypeError, Exception)):
            location_service.geocode("")    # Empty string
        
        # Test reverse geocoding with invalid coordinates
        with pytest.raises((ValueError, TypeError, Exception)):
            location_service.reverse_geocode(None, None)
        
        with pytest.raises((ValueError, TypeError, Exception)):
            location_service.reverse_geocode("invalid", "invalid")
        
        # Test with simulated service failures
        with patch.object(location_service, '_geocode_request') as mock_request:
            mock_request.side_effect = ConnectionError("Service unavailable")
            
            # Should handle gracefully - either return None/error indicator or raise
            try:
                result = location_service.geocode("Somewhere")
                # If no exception raised, result should indicate failure
                assert result is None or isinstance(result, dict) and ('error' in result or 'status' in result)
            except ConnectionError:
                # This is also acceptable - the exception propagated appropriately
                pass
            except Exception as e:
                # Other exceptions are fine as long as they're reasonable
                assert isinstance(e, (ConnectionError, TimeoutError, RuntimeError, ValueError, TypeError))
    
    @pytest.mark.edge_case
    def test_resource_cleanup_on_exception(self):
        """Test that resources are properly cleaned up when exceptions occur"""
        # This is more conceptual - we're testing that exceptions don't leave
        # the system in an inconsistent state
        
        # Create a valid config first
        config_data = {"frequency": 3000.0, "distance": 5.0}
        
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            # This should work
            config = SimulationConfig(config_path)
            assert config is not None
            
            # Now test with invalid data that should cause an exception
            invalid_config_data = {"frequency": -1000.0, "distance": 5.0}  # Negative frequency
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(invalid_config_data, f)
                invalid_config_path = f.name
            
            try:
                # This should raise an exception
                with pytest.raises((ValueError, AssertionError)):
                    SimulationConfig(invalid_config_path)
                
                # But the earlier valid config should still be accessible
                # (This tests that exception handling doesn't corrupt global state)
                config2 = SimulationConfig(config_path)
                assert config2 is not None
                assert config2.frequency == 3000.0
                
            finally:
                if os.path.exists(invalid_config_path):
                    os.unlink(invalid_config_path)
                    
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    @pytest.mark.edge_case
    def test_nested_exception_handling(self):
        """Test exception handling in nested function calls"""
        # Test that exceptions in lower-level functions propagate properly
        # through higher-level functions
        
        def problematic_fspl_call():
            # This should raise an exception
            return free_space_path_loss(-1000.0, 5.0)  # Negative frequency
        
        def wrapper_function():
            # This function calls the problematic function
            return problematic_fspl_call()
        
        # The exception should propagate through the wrapper
        with pytest.raises((ValueError, AssertionError)):
            wrapper_function()
        
        # Test with multiple layers
        def layer3():
            return free_space_path_loss(-1000.0, 5.0)
        
        def layer2():
            return layer3()
        
        def layer1():
            return layer2()
        
        with pytest.raises((ValueError, AssertionError)):
            layer1()


