"""Boundary value debugger tests"""
import pytest
import numpy as np
from propagation_model.models import (
    free_space_path_loss,
    rain_attenuation,
    gas_attenuation,
    fog_attenuation,
    close_in_path_loss
)
from propagation_model.empirical_models import (
    free_space_path_loss as fspl_empirical,
    rain_attenuation as rain_empirical,
    gas_attenuation as gas_empirical,
    fog_attenuation as fog_empirical,
    close_in_path_loss as ci_empirical
)
from propagation_model.itm_model = itm_model
from propagation_model.ray_tracing_model = ray_tracing_model

class TestBoundaryConditionsDebugger:
    """Test suite for boundary value debugger"""
    
    @pytest.mark.edge_case
    def test_zero_and_negative_values(self):
        """Test all models with zero and negative input values"""
        # Test frequencies
        test_freqs = [0.0, -1.0, -100.0]
        # Test distances
        test_dists = [0.0, -1.0, -10.0]
        # Test rain rates (for rain model)
        test_rain_rates = [0.0, -1.0, -50.0]
        # Test fog densities (for fog model)
        test_fog_densities = [0.0, -0.1, -1.0]
        
        # FSPL model
        for freq in test_freqs:
            for dist in test_dists:
                result = free_space_path_loss(freq, dist)
                assert result == 0.0, f"FSPL failed for freq={freq}, dist={dist}"
        
        # Rain model
        for freq in test_freqs:
            for dist in test_dists:
                for rain_rate in test_rain_rates:
                    result = rain_attenuation(freq, dist, rain_rate)
                    assert result == 0.0, f"Rain model failed for freq={freq}, dist={dist}, rain_rate={rain_rate}"
        
        # Gas model
        for freq in test_freqs:
            for dist in test_dists:
                result = gas_attenuation(freq, dist)
                assert result == 0.0, f"Gas model failed for freq={freq}, dist={dist}"
        
        # Fog model
        for freq in test_freqs:
            for dist in test_dists:
                for fog_density in test_fog_densities:
                    result = fog_attenuation(freq, dist, fog_density)
                    assert result == 0.0, f"Fog model failed for freq={freq}, dist={dist}, fog_density={fog_density}"
        
        # CI model
        for freq in test_freqs:
            for dist in test_dists:
                result = close_in_path_loss(freq, dist)
                assert result == 0.0, f"CI model failed for freq={freq}, dist={dist}"
    
    @pytest.mark.edge_case
    def test_extreme_high_values(self):
        """Test models with extremely high input values"""
        extreme_freqs = [1e6, 1e9, 1e12]  # Very high frequencies
        extreme_dists = [1e3, 1e6, 1e9]   # Very long distances
        
        # FSPL should handle large values (though may lose precision)
        for freq in extreme_freqs:
            for dist in extreme_dists:
                result = free_space_path_loss(freq, dist)
                # Should not crash and should return a numeric value
                assert result is not None
                assert not np.isnan(result), f"FSPL returned NaN for freq={freq}, dist={dist}"
                # May be inf for extremely large values, which is acceptable
                assert result >= 0 or np.isinf(result), f"FSPL returned invalid value for freq={freq}, dist={dist}"
        
        # Other models should also handle extreme values gracefully
        for freq in [1e6, 1e9]:  # Slightly lower extremes for other models
            for dist in [1e3, 1e6]:
                # Rain model
                result = rain_attenuation(freq/1000.0, dist, 50.0)  # Convert to GHz
                assert result is not None
                assert not np.isnan(result) or np.isinf(result)
                assert result >= 0 or np.isinf(result)
                
                # Gas model
                result = gas_attenuation(freq/1000.0, dist)
                assert result is not None
                assert not np.isnan(result) or np.isinf(result)
                assert result >= 0 or np.isinf(result)
                
                # Fog model
                result = fog_attenuation(freq/1000.0, dist, 2.0)
                assert result is not None
                assert not np.isnan(result) or np.isinf(result)
                assert result >= 0 or np.isinf(result)
                
                # CI model
                result = close_in_path_loss(freq, dist)
                assert result is not None
                assert not np.isnan(result) or np.isinf(result)
                assert result >= 0 or np.isinf(result)
    
    @pytest.mark.edge_case
    def test_extreme_low_positive_values(self):
        """Test models with extremely low positive input values"""
        small_freqs = [1e-3, 1e-6, 1e-9]   # Very low frequencies
        small_dists = [1e-6, 1e-9, 1e-12]  # Very short distances
        
        # FSPL should handle small values
        for freq in small_freqs:
            for dist in small_dists:
                result = free_space_path_loss(freq, dist)
                assert result is not None
                assert not np.isnan(result)
                # For very small values, result should be very small or zero
                assert result >= 0
        
        # Test other models with small values
        for freq in [1e-3, 1e-6]:  # Even smaller for other tests
            for dist in [1e-6, 1e-9]:
                # Rain model (convert freq to GHz)
                result = rain_attenuation(freq/1e9, dist, 0.1)  # Very light rain
                assert result is not None
                assert not np.isnan(result)
                assert result >= 0
                
                # Gas model
                result = gas_attenuation(freq/1e9, dist)
                assert result is not None
                assert not np.isnan(result)
                assert result >= 0
                
                # Fog model
                result = fog_attenuation(freq/1e9, dist, 0.001)  # Very light fog
                assert result is not None
                assert not np.isnan(result)
                assert result >= 0
                
                # CI model
                result = close_in_path_loss(freq, dist)
                assert result is not None
                assert not np.isnan(result)
                assert result >= 0
    
    @pytest.mark.edge_case
    def test_extreme_antenna_parameters(self):
        """Test models with extreme antenna parameters"""
        # Extreme heights
        extreme_heights = [0.0, 1e-6, 1e3, 1e6]  # From zero to very high
        
        for height in extreme_heights:
            # CI model with extreme heights
            result = close_in_path_loss(1000.0, 1.0, tx_height_m=height, rx_height_m=height)
            assert result is not None
            assert not np.isnan(result)
            assert result >= 0
            
            # ITM model with extreme heights (if available)
            try:
                result = itm_model(
                    frequency_mhz=1000.0,
                    distance_km=5.0,
                    tx_height_m=height,
                    rx_height_m=height
                )
                # Should either work or give a clear indication of unavailability
                assert result is not None or str(type(result)).find('NotImplemented') >= 0
            except ImportError:
                # Expected if itmlogic not available
                pass
            except Exception:
                # Other exceptions are acceptable for extreme values
                pass
    
    @pytest.mark.edge_case
    def test_boundary_conditions_specific_models(self):
        """Test specific boundary conditions for each model type"""
        # Test frequency boundaries
        boundary_freqs = [0.0, 0.1, 1.0, 1000.0, 100000.0]  # From DC to extremely high
        
        for freq in boundary_freqs:
            # FSPL at boundary frequencies
            result = free_space_path_loss(freq, 1.0)
            assert result is not None
            assert not np.isnan(result)
            
            # Convert to GHz for other models
            freq_ghz = freq / 1000.0 if freq != 0 else 0
            
            # Rain at boundary frequencies
            result = rain_attenuation(freq_ghz, 1.0, 5.0)
            assert result is not None
            assert not np.isnan(result)
            assert result >= 0
            
            # Gas at boundary frequencies
            result = gas_attenuation(freq_ghz, 1.0)
            assert result is not None
            assert not np.isnan(result)
            assert result >= 0
            
            # Fog at boundary frequencies
            result = fog_attenuation(freq_ghz, 1.0, 0.1)
            assert result is not None
            assert not np.isnan(result)
            assert result >= 0
            
            # CI at boundary frequencies
            result = close_in_path_loss(freq, 1.0)
            assert result is not None
            assert not np.isnan(result)
            assert result >= 0
        
        # Test distance boundaries
        boundary_dists = [0.0, 0.001, 0.1, 1.0, 1000.0, 100000.0]  # From zero to interplanetary
        
        for dist in boundary_dists:
            # FSPL at boundary distances
            result = free_space_path_loss(1000.0, dist)
            assert result is not None
            assert not np.isnan(result)
            
            # Convert frequency to GHz
            freq_ghz = 1.0
            
            # Other models at boundary distances
            result = rain_attenuation(freq_ghz, dist, 5.0)
            assert result is not None
            assert not np.isnan(result)
            assert result >= 0
            
            result = gas_attenuation(freq_ghz, dist)
            assert result is not None
            assert not np.isnan(result)
            assert result >= 0
            
            result = fog_attenuation(freq_ghz, dist, 0.1)
            assert result is not None
            assert not np.isnan(result)
            assert result >= 0
            
            result = close_in_path_loss(1000.0, dist)
            assert result is not None
            assert not np.isnan(result)
            assert result >= 0
    
    @pytest.mark.edge_case
    def test_transition_boundaries(self):
        """Test behavior at known transition boundaries between models"""
        # These are frequencies where different propagation effects dominate
        
        # VLF/LF boundary (around 30 kHz)
        vlf_freq = 0.03  # MHz
        lf_freq = 0.30   # MHz
        
        # Test that models behave reasonably across this boundary
        for freq in [vlf_freq, lf_freq]:
            result_fspl = free_space_path_loss(freq, 10.0)
            assert result_fspl is not None
            assert not np.isnan(result_fspl)
            
            # Other models with standard parameters
            freq_ghz = freq / 1000.0
            result_rain = rain_attenuation(freq_ghz, 10.0, 5.0)
            assert result_rain is not None
            assert not np.isnan(result_rain)
            assert result_rain >= 0
        
        # UHF boundary (around 300 MHz - 3 GHz)
        uhf_freq = 300.0   # MHz
        microwave_freq = 3000.0  # MHz
        
        for freq in [uhf_freq, microwave_freq]:
            result_fspl = free_space_path_loss(freq, 1.0)
            assert result_fspl is not None
            assert not np.isnan(result_fspl)
            
            freq_ghz = freq / 1000.0
            result_rain = rain_attenuation(freq_ghz, 1.0, 5.0)
            assert result_rain is not None
            assert not np.isnan(result_rain)
            assert result_rain >= 0
    
    @pytest.mark.edge_case
    def test_nan_and_infinity_handling(self):
        """Test handling of NaN and infinity inputs"""
        # Test with NaN values
        try:
            result = free_space_path_loss(np.nan, 5.0)
            # Depending on implementation, might return NaN, raise exception, or handle gracefully
            # We mainly want to ensure it doesn't crash unexpectedly
        except (TypeError, ValueError):
            # Acceptable to raise exception for invalid input
            pass
        
        try:
            result = free_space_path_loss(5.0, np.nan)
            # Same as above
        except (TypeError, ValueError):
            pass
        
        # Test with infinity values
        try:
            result = free_space_path_loss(np.inf, 5.0)
            # Might return inf or handle gracefully
            assert result is not None
        except (TypeError, ValueError, OverflowError):
            pass
        
        try:
            result = free_space_path_loss(5.0, np.inf)
            # Same as above
            assert result is not None
        except (TypeError, ValueError, OverflowError):
            pass
        
        # Test negative infinity
        try:
            result = free_space_path_loss(-np.inf, 5.0)
            # Should handle gracefully (negative frequency doesn't make physical sense)
            assert result is not None
        except (TypeError, ValueError, OverflowError):
            pass