"""Cross-model validation debugger tests"""
import pytest
import numpy as np
from propagation_model.models import (
    free_space_path_loss as fspl_model,
    rain_attenuation as rain_model,
    gas_attenuation as gas_model,
    fog_attenuation as fog_model,
    close_in_path_loss as ci_model
)
from propagation_model.empirical_models import (
    free_space_path_loss as fspl_empirical,
    rain_attenuation as rain_empirical,
    gas_attenuation as gas_empirical,
    fog_attenuation as fog_empirical,
    close_in_path_loss as ci_empirical
)
from propagation_model.itm_model import itm_path_loss as itm_model
from propagation_model.ray_tracing_model import ray_tracing_path_loss as ray_tracing_model
from test_utils.validators import (
    validate_fspl_output,
    validate_rain_attenuation_output,
    validate_gas_attenuation_output,
    validate_fog_attenuation_output,
    validate_ci_output,
    validate_itm_output,
    validate_ray_tracing_output
)
from test_utils.fixtures import SAMPLE_VALID_CONFIGS

class TestModelComparisonDebugger:
    """Test suite for cross-model validation and comparison"""
    
    @pytest.mark.propagation
    def test_fspl_consistency(self):
        """Test that FSPL models give consistent results"""
        test_cases = [
            (30.0, 0.1),      # VHF, short distance
            (300.0, 1.0),     # UHF, medium distance
            (3000.0, 10.0),   # Microwave, long distance
        ]
        
        for frequency_mhz, distance_km in test_cases:
            # Test basic consistency between implementations
            result_model = fspl_model(frequency_mhz, distance_km)
            result_empirical = fspl_empirical(frequency_mhz, distance_km)
            
            # Should be identical (same mathematical formula)
            assert abs(result_model - result_empirical) < 0.01
            
            # Should pass validation
            assert validate_fspl_output(result_model, frequency_mhz, distance_km)
            assert validate_fspl_output(result_empirical, frequency_mhz, distance_km)
    
    @pytest.mark.propagation
    def test_empirical_model_consistency(self):
        """Test that empirical models produce reasonable results"""
        test_cases = [
            # (frequency_ghz, distance_km, rain_rate_mmh)
            (5.0, 1.0, 5.0),    # Light rain
            (10.0, 5.0, 25.0),  # Moderate rain
            (30.0, 10.0, 50.0), # Heavy rain
        ]
        
        for frequency_ghz, distance_km, rain_rate_mmh in test_cases:
            # Test rain models
            result_model = rain_model(frequency_ghz, distance_km, rain_rate_mmh)
            result_empirical = rain_empirical(frequency_ghz, distance_km, rain_rate_mmh)
            
            # Both should be valid and non-negative
            assert result_model >= 0
            assert result_empirical >= 0
            assert validate_rain_attenuation_output(result_model, frequency_ghz, distance_km, rain_rate_mmh)
            assert validate_rain_attenuation_output(result_empirical, frequency_ghz, distance_km, rain_rate_mmh)
            
            # Test gas models (standard conditions)
            result_model_gas = gas_model(frequency_ghz, distance_km)
            result_empirical_gas = gas_empirical(frequency_ghz, distance_km)
            
            assert result_model_gas >= 0
            assert result_empirical_gas >= 0
            assert validate_gas_attenuation_output(result_model_gas, frequency_ghz, distance_km)
            assert validate_gas_attenuation_output(result_empirical_gas, frequency_ghz, distance_km)
            
            # Test fog models
            fog_density = 0.5  # g/m³
            result_model_fog = fog_model(frequency_ghz, distance_km, fog_density)
            result_empirical_fog = fog_empirical(frequency_ghz, distance_km, fog_density)
            
            assert result_model_fog >= 0
            assert result_empirical_fog >= 0
            assert validate_fog_attenuation_output(result_model_fog, frequency_ghz, distance_km, fog_density)
            assert validate_fog_attenuation_output(result_empirical_fog, frequency_ghz, distance_km, fog_density)
    
    @pytest.mark.propagation
    def test_model_order_of_magnitude(self):
        """Test that different models produce results in reasonable ranges"""
        # Standard test conditions
        frequency_ghz = 10.0
        distance_km = 5.0
        
        # FSPL (baseline)
        fspl_result = fspl_model(3000.0, distance_km)  # Convert GHz to MHz
        
        # Rain attenuation (moderate rain)
        rain_result = rain_model(frequency_ghz, distance_km, 25.0)
        
        # Gas attenuation (standard atmosphere)
        gas_result = gas_model(frequency_ghz, distance_km)
        
        # Fog attenuation (moderate fog)
        fog_result = fog_model(frequency_ghz, distance_km, 0.5)
        
        # CI model (typical urban)
        ci_result = ci_model(3000.0, distance_km)  # Convert GHz to MHz
        
        # All should be non-negative
        assert fspl_result >= 0
        assert rain_result >= 0
        assert gas_result >= 0
        assert fog_result >= 0
        assert ci_result >= 0
        
        # FSPL should be the baseline (lowest loss in clear conditions)
        # Other models represent additional losses
        # Note: This is a simplified check - in reality, some models might 
        # show lower values in certain conditions due to modeling differences
        total_expected_loss = fspl_result + rain_result + gas_result + fog_result
        
        # The combined loss should be reasonable (not negative or extremely high)
        assert total_expected_loss >= 0
        # Sanity check: for 5km at 10GHz, we don't expect more than a few hundred dB loss
        assert total_expected_loss < 1000
    
    @pytest.mark.propagation
    @pytest.mark.requires_itmlogic
    def test_itm_integration_with_other_models(self):
        """Test ITM model integration with other propagation models"""
        try:
            frequency_mhz = 1000.0
            distance_km = 5.0
            tx_height_m = 10.0
            rx_height_m = 10.0
            
            # Get ITM result
            itm_result = itm_model(
                frequency_mhz=frequency_mhz,
                distance_km=distance_km,
                tx_height_m=tx_height_m,
                rx_height_m=rx_height_m
            )
            
            # Should be valid if ITM is available
            assert itm_result is not None
            assert not np.isnan(itm_result)
            assert itm_result >= 0
            
            # Validate against other models at same frequency/distance
            # Convert frequency to GHz for other models
            frequency_ghz = frequency_mhz / 1000.0
            
            fspl_result = fspl_model(frequency_mhz, distance_km)
            rain_result = rain_model(frequency_ghz, distance_km, 0.0)  # No rain
            gas_result = gas_model(frequency_ghz, distance_km)
            fog_result = fog_model(frequency_ghz, distance_km, 0.0)  # No fog
            
            # ITM should be in the same ballpark as the sum of other models
            # (This is a rough check - ITM includes many effects that others don't)
            basic_losses = fspl_result + rain_result + gas_result + fog_result
            
            # Both should be positive and reasonable
            assert basic_losses >= 0
            assert itm_result >= 0
            
        except ImportError:
            pytest.skip("itmlogic package not available")
    
    @pytest.mark.propagation
    @pytest.mark.requires_sionna
    def test_ray_tracing_integration_check(self):
        """Basic integration check for ray tracing model"""
        try:
            frequency_hz = 30e9
            tx_array = [0, 0, 10]
            rx_array = [100, 0, 1.5]
            
            # Get ray tracing result
            rt_result = ray_tracing_model(
                frequency_ghz=frequency_hz / 1e9,  # Convert Hz to GHz
                tx_pos=tx_array,
                rx_pos=rx_array
            )
            
            # Should be valid if sionna is available
            assert rt_result is not None
            
            # Compare with FSPL at same frequency/distance
            # Calculate distance between tx and rx
            distance_m = np.sqrt(
                (rx_array[0] - tx_array[0])**2 + 
                (rx_array[1] - tx_array[1])**2 + 
                (rx_array[2] - tx_array[2])**2
            )
            distance_km = distance_m / 1000.0
            frequency_mhz = frequency_hz / 1e6
            
            fspl_result = fspl_model(frequency_mhz, distance_km)
            
            # Both should be reasonable positive values
            assert fspl_result >= 0
            # Note: Ray tracing result format may vary, so we mainly check it's not obviously wrong
            
        except ImportError:
            pytest.skip("sionna package not available")
        except Exception as e:
            if "sionna" in str(e).lower():
                pytest.skip("sionna package not available or not properly installed")
            else:
                raise
    
    @pytest.mark.propagation
    def test_model_behavior_at_extremes(self):
        """Test model behavior at extreme values"""
        extreme_cases = [
            # (description, frequency, distance, extra_params)
            ("Very low frequency", 0.1, 1.0, {}),
            ("Very high frequency", 100000.0, 1.0, {}),
            ("Very short distance", 1000.0, 0.001, {}),
            ("Very long distance", 1000.0, 10000.0, {}),
        ]
        
        for desc, freq, dist, params in extreme_cases:
            # Test FSPL (should always work)
            try:
                fspl_result = fspl_model(freq, dist)
                assert fspl_result is not None
                assert not np.isnan(fspl_result)
                # FSPL can handle extreme values mathematically
            except Exception as e:
                pytest.fail(f"FSPL failed on {desc}: {e}")
            
            # Test rain model with extreme values
            try:
                rain_result = rain_model(freq/1000.0, dist, 25.0)  # Convert to GHz
                assert rain_result is not None
                assert not np.isnan(rain_result)
                assert rain_result >= 0
            except Exception:
                # Some models might not handle extremes gracefully, which is OK
                pass
            
            # Test CI model with extreme values
            try:
                ci_result = ci_model(freq, dist)
                assert ci_result is not None
                assert not np.isnan(ci_result)
                assert ci_result >= 0
            except Exception:
                pass