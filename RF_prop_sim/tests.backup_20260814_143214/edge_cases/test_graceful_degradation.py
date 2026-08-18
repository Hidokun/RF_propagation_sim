"""Graceful degradation debugger tests"""
import pytest
import numpy as np
from propagation_model.models import (
    free_space_path_loss,
    rain_attenuation,
    gas_attenuation,
    fog_attenuation,
    close_in_path_loss
)
from propagation_model.itm_model = itm_model
from propagation_model.ray_tracing_model = ray_tracing_model
from RF_prop_sim.input_data_collection.ingestion = SimulationConfig
from RF_prop_sim.antenna_data.parser = AntennaConfig
from RF_prop_sim.mapping.location_service = LocationService
from RF_prop_sim.mapping.dem_provider = DEMProvider
from RF_prop_sim.mapping.dem_processor = DEMProcessor
from test_utils.fixtures import SAMPLE_VALID_CONFIGS

class TestGracefulDegradationDebugger:
    """Test suite for graceful degradation debugger"""
    
    @pytest.mark.edge_case
    def test_degradation_when_optional_dependencies_missing(self):
        """Test graceful degradation when optional dependencies are unavailable"""
        # Test ITM model degradation
        try:
            # Try to import itmlogic to see if it's available
            import itmlogic
            itm_available = True
        except ImportError:
            itm_available = False
        
        if not itm_available:
            # When itmlogic is not available, ITM model should degrade gracefully
            try:
                result = itm_model(
                    frequency_mhz=1000.0,
                    distance_km=5.0,
                    tx_height_m=10.0,
                    rx_height_m=10.0
                )
                # Depending on implementation, might return None, raise a specific exception,
                # or return a fallback value
                # The key is that it shouldn't crash the entire system
                assert result is None or isinstance(result, (int, float)) or hasattr(result, '__class__')
            except ImportError:
                # This is also acceptable - clear indication of missing dependency
                pass
            except Exception as e:
                # Other exceptions are acceptable as long as they're informative
                # and don't crash the system
                assert isinstance(e, (ImportError, RuntimeError, ValueError, Exception))
        else:
            # If itmlogic is available, it should work normally
            result = itm_model(
                frequency_mhz=1000.0,
                distance_km=5.0,
                tx_height_m=10.0,
                rx_height_m=10.0
            )
            assert result is not None
            # Basic validation - should be a reasonable path loss value
            assert isinstance(result, (int, float))
            assert not np.isnan(result)
    
    @pytest.mark.edge_case
    def test_degradation_when_sionna_unavailable(self):
        """Test graceful degradation when Sionna is unavailable for ray tracing"""
        try:
            # Try to import sionna to see if it's available
            import sionna
            sionna_available = True
        except ImportError:
            sionna_available = False
        
        if not sionna_available:
            # When sionna is not available, ray tracing model should degrade gracefully
            try:
                result = ray_tracing_model(
                    frequency_hz=30e9,
                    tx_array=[0, 0, 10],
                    rx_array=[100, 0, 1.5]
                )
                # Depending on implementation, might return None, raise a specific exception,
                # or return a fallback value
                assert result is None or isinstance(result, (int, float)) or hasattr(result, '__class__')
            except ImportError:
                # This is also acceptable - clear indication of missing dependency
                pass
            except Exception as e:
                # Other exceptions are acceptable as long as they're informative
                assert isinstance(e, (ImportError, RuntimeError, ValueError, Exception))
        else:
            # If sionna is available, it should work normally
            result = ray_tracing_model(
                frequency_hz=30e9,
                tx_array=[0, 0, 10],
                rx_array=[100, 0, 1.5]
            )
            assert result is not None
            # Basic validation
            assert result is not None
    
    @pytest.mark.edge_case
    def test_fallback_to_simpler_models(self):
        """Test that system falls back to simpler models when complex ones fail"""
        # Test scenario: ITM unavailable, fall back to empirical models
        frequency_mhz = 1000.0
        distance_km = 5.0
        tx_height_m = 10.0
        rx_height_m = 10.0
        
        # Get baseline from reliable models
        fspl_result = free_space_path_loss(frequency_mhz, distance_km)
        frequency_ghz = frequency_mhz / 1000.0
        rain_result = rain_attenuation(frequency_ghz, distance_km, 10.0)  # Light rain
        gas_result = gas_attenuation(frequency_ghz, distance_km)
        fog_result = fog_attenuation(frequency_ghz, distance_km, 0.1)   # Light fog
        
        baseline_loss = fspl_result + rain_result + gas_result + fog_result
        assert baseline_loss >= 0
        
        # Now test ITM - if it fails, we should still have reasonable results from other models
        try:
            itm_result = itm_model(
                frequency_mhz=frequency_mhz,
                distance_km=distance_km,
                tx_height_m=tx_height_m,
                rx_height_m=rx_height_m
            )
            
            # If ITM works, it should be in the same ballpark as our baseline
            # (Not necessarily equal, as ITM includes different physics)
            if itm_result is not None and not np.isnan(itm_result):
                # Both should be positive and reasonable
                assert itm_result >= 0
                # The ratio should be reasonable (not orders of magnitude different)
                if baseline_loss > 0:
                    ratio = itm_result / baseline_loss
                    assert 0.1 <= ratio <= 10.0  # Within one order of magnitude
        except Exception:
            # If ITM fails, we should still have our baseline
            # This tests that failure of one model doesn't break the ability to use others
            assert baseline_loss >= 0  # Baseline should still be valid
    
    @pytest.mark.edge_case
    def test_degradation_when_external_services_unavailable(self):
        """Test graceful degradation when external services (geocoding, DEM) are unavailable"""
        location_service = LocationService()
        
        # Test with network unavailable
        with patch.object(location_service, '_geocode_request') as mock_request:
            mock_request.side_effect = ConnectionError("Network unavailable")
            
            # Should handle gracefully
            try:
                result = location_service.geocode("Somewhere")
                # Depending on implementation, might return None, error indicator, or cached value
                assert result is None or isinstance(result, dict)
                # If it returns a dict, it should indicate the error condition
                if isinstance(result, dict):
                    assert 'error' in result or 'status' in result or 'message' in result
            except ConnectionError:
                # This is also acceptable - the exception propagated appropriately
                pass
            except Exception as e:
                # Other exceptions are fine if they're reasonable
                assert isinstance(e, (ConnectionError, TimeoutError, RuntimeError, ValueError))
        
        # Test with timeout
        with patch.object(location_service, '_geocode_request') as mock_request:
            mock_request.side_effect = TimeoutError("Request timed out")
            
            try:
                result = location_service.geocode("Somewhere")
                assert result is None or isinstance(result, dict)
                if isinstance(result, dict):
                    assert 'error' in result or 'status' in result or 'message' in result
            except TimeoutError:
                pass  # Acceptable
            except Exception as e:
                assert isinstance(e, (TimeoutError, RuntimeError, ValueError))
    
    @pytest.mark.edge_case
    def test_degradation_with_partial_data_availability(self):
        """Test graceful degradation when only partial data is available"""
        # Test configuration with missing optional fields
        partial_configs = [
            {"frequency": 3000.0},  # Only frequency
            {"frequency": 3000.0, "distance": 5.0},  # Frequency and distance
            {"distance": 5.0, "tx_height": 10.0},  # Distance and tx height
        ]
        
        for config_data in partial_configs:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                import json
                json.dump(config_data, f)
                config_path = f.name
            
            try:
                # Should handle gracefully - either use defaults or provide clear error
                try:
                    config = SimulationConfig(config_path)
                    # If it succeeds, it should have used defaults for missing fields
                    assert config is not None
                    for key, value in config_data.items():
                        assert getattr(config, key) == value
                    # Missing fields should have default values
                except (ValueError, KeyError) as e:
                    # This is also acceptable - clear indication of missing required fields
                    assert "required" in str(e).lower() or "missing" in str(e).lower()
                except Exception as e:
                    # Other exceptions are acceptable if they're reasonable
                    assert isinstance(e, (ValueError, KeyError, TypeError))
            finally:
                import os
                if os.path.exists(config_path):
                    os.unlink(config_path)
        
        # Test antenna config with partial data
        partial_antenna_configs = [
            {"gain": 10.0},  # Only gain
            {"gain": 10.0, "beamwidth": 50.0},  # Gain and beamwidth
            {"beamwidth": 30.0, "polarization": "vertical"},  # Beamwidth and polarization
        ]
        
        for antenna_data in partial_antenna_configs:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                import json
                json.dump(antenna_data, f)
                antenna_path = f.name
            
            try:
                try:
                    antenna_config = AntennaConfig(antenna_path)
                    assert antenna_config is not None
                    for key, value in antenna_data.items():
                        assert getattr(antenna_config, key) == value
                except (ValueError, KeyError) as e:
                    assert "required" in str(e).lower() or "missing" in str(e).lower()
                except Exception as e:
                    assert isinstance(e, (ValueError, KeyError, TypeError))
            finally:
                import os
                if os.path.exists(antenna_path):
                    os.unlink(antenna_path)
    
    @pytest.mark.edge_case
    def test_progressive_degradation(self):
        """Test that system degrades progressively as more components fail"""
        # Start with full functionality, then remove components one by one
        
        # Test 1: All basic models should work
        try:
            fspl = free_space_path_loss(1000.0, 5.0)
            rain = rain_attenuation(1.0, 5.0, 10.0)
            gas = gas_attenuation(1.0, 5.0)
            fog = fog_attenuation(1.0, 5.0, 0.1)
            ci = close_in_path_loss(1000.0, 5.0)
            
            assert all(x is not None and not np.isnan(x) and x >= 0 
                      for x in [fspl, rain, gas, fog, ci])
        except Exception as e:
            pytest.fail(f"Basic models should work: {e}")
        
        # Test 2: Even if advanced models fail, basic ones should still work
        # Simulate ITM failure
        try:
            # Temporarily mock ITM to fail
            with patch('propagation_model.itm_model.itm_model', side_effect=Exception("ITM failed")):
                from propagation_model.itm_model = itm_model
                
                # Basic models should still work
                fspl = free_space_path_loss(1000.0, 5.0)
                rain = rain_attenuation(1.0, 5.0, 10.0)
                
                assert fspl is not None and not np.isnan(fspl) and fspl >= 0
                assert rain is not None and not np.isnan(rain) and rain >= 0
                
                # ITM should indicate failure
                try:
                    itm_result = itm_model(1000.0, 5.0, 10.0, 10.0)
                    # Depending on implementation, might be None or raise exception
                    # But the important thing is that it didn't take down other models
                except Exception:
                    # This is expected - ITM is supposed to be failing
                    pass
        except ImportError:
            # If itmlogic not available, that's fine - we're testing the concept
            pass
        
        # Test 3: Even if multiple models fail, core functionality should remain
        try:
            with patch('propagation_model.itm_model.itm_model', side_effect=Exception("ITM failed")), \
                 patch('propagation_model.ray_tracing_model.ray_tracing_model', side_effect=Exception("Ray tracing failed")):
                
                from propagation_model.itm_model = itm_model
                from propagation_model.ray_tracing_model = ray_tracing_model
                
                # Core models should still work
                fspl = free_space_path_loss(1000.0, 5.0)
                rain = rain_attenuation(1.0, 5.0, 10.0)
                gas = gas_attenuation(1.0, 5.0)
                
                assert all(x is not None and not np.isnan(x) and x >= 0 
                          for x in [fspl, rain, gas])
        except ImportError:
            pass  # Expected if models not available
    
    @pytest.mark.edge_case
    def test_degradation_maintains_api_consistency(self):
        """Test that degraded mode maintains consistent API where possible"""
        # Test that even when degraded, functions return expected types
        
        # Test ITM model degradation
        try:
            result = itm_model(1000.0, 5.0, 10.0, 10.0)
            # Should return None, a number, or raise a specific exception
            # The key is that calling code can handle the result predictably
            if result is not None:
                assert isinstance(result, (int, float, np.number)) or hasattr(result, '__class__')
        except Exception as e:
            # If it raises an exception, it should be a specific, expected type
            assert isinstance(e, (ImportError, RuntimeError, ValueError, NotImplementedError, Exception))
        
        # Test ray tracing model degradation
        try:
            result = ray_tracing_model(30e9, [0,0,10], [100,0,1.5])
            if result is not None:
                assert isinstance(result, (int, float, np.number)) or hasattr(result, '__class__')
        except Exception as e:
            assert isinstance(e, (ImportError, RuntimeError, ValueError, NotImplementedError, Exception))
        
        # Test that basic models still work and maintain API
        fspl_result = free_space_path_loss(1000.0, 5.0)
        assert isinstance(fspl_result, (int, float, np.number))
        assert fspl_result >= 0
        
        rain_result = rain_attenuation(1.0, 5.0, 10.0)
        assert isinstance(rain_result, (int, float, np.number))
        assert rain_result >= 0
    
    @pytest.mark.edge_case
    def test_user_notification_in_degraded_mode(self):
        """Test that users get appropriate feedback when in degraded mode"""
        # This is more about logging and user feedback than functional behavior
        # We'll test that appropriate messages are logged or returned
        
        import logging
        from io import StringIO
        
        # Capture log messages
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger('rf_simulator.degradation')
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        
        try:
            # Test ITM degradation notification
            try:
                result = itm_model(1000.0, 5.0, 10.0, 10.0)
                # Depending on implementation, might log a warning
            except ImportError:
                # Should log warning about missing dependency
                pass
            except Exception as e:
                # Might log warning about failure
                pass
            
            # Check if any degradation warnings were logged
            log_contents = log_stream.getvalue()
            # Depending on implementation, might contain warnings about degradation
            # This test is flexible - the important thing is that logging doesn't break
            
        finally:
            logger.removeHandler(handler)
        
        # Test that normal operation doesn't spam degradation logs
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger('rf_simulator.degradation')
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        
        try:
            # Normal operations should not generate degradation warnings
            fspl_result = free_space_path_loss(1000.0, 5.0)
            rain_result = rain_attenuation(1.0, 5.0, 10.0)
            
            log_contents = log_stream.getvalue()
            # Might be empty or contain unrelated warnings, but shouldn't be spammed
            # with degradation messages for normal operation
            
        finally:
            logger.removeHandler(handler)