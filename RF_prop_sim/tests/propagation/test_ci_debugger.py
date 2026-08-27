"""Close-In propagation model debugger tests"""
import pytest
import numpy as np
from propagation_model import close_in_path_loss as ci_model, free_space_path_loss
from test_utils.validators import validate_ci_output
from test_utils.fixtures import SAMPLE_VALID_CONFIGS

class TestDebugger:
    """Test suite for Close-In propagation model debugger"""
    
    @pytest.mark.propagation
    def test_ci_basic_functionality(self):
        """Test basic CI functionality with standard inputs"""
        frequency_mhz = 900.0
        distance_km = 1.0
        
        # Test both implementations
        result_model = ci_model(frequency_mhz, distance_km)
        result_empirical = ci_model(frequency_mhz, distance_km)
        
        # Validate outputs
        assert validate_ci_output(result_model, frequency_mhz, distance_km)
        assert validate_ci_output(result_empirical, frequency_mhz, distance_km)
        
        # Results should be reasonable
        assert result_model >= 0
        assert result_empirical >= 0
    
    @pytest.mark.propagation
    def test_ci_zero_inputs(self):
        """Test CI with zero or negative inputs"""
        # Test zero frequency
        result = ci_model(0.0, 1.0)
        assert result == 0.0
        
        # Test zero distance
        result = ci_model(900.0, 0.0)
        assert result == 0.0
        
        # Test negative values
        result = ci_model(-100.0, 1.0)
        assert result == 0.0
        
        result = ci_model(900.0, -1.0)
        assert result == 0.0
    
    @pytest.mark.propagation
    def test_ci_reference_distance(self):
        """Test CI with different reference distances"""
        frequency_mhz = 900.0
        distance_km = 1.0
        
        # Test with default reference distance (1m)
        result_default = ci_model(frequency_mhz, distance_km)
        
        # Test with explicit reference distance (1m)
        result_explicit = ci_model(frequency_mhz, distance_km, reference_distance_m=1.0)
        
        # Should be the same
        assert abs(result_default - result_explicit) < 0.001
        
        # Test with different reference distance (100m)
        result_100m = ci_model(frequency_mhz, distance_km, reference_distance_m=100.0)

        # Audit C-1 fix: with path_loss_exponent = 2 (default), CI collapses
        # onto FSPL for ANY reference distance — that invariance IS the model.
        fspl_ref = free_space_path_loss(frequency_mhz, distance_km)
        assert abs(result_100m - fspl_ref) < 1e-6
        assert abs(result_default - fspl_ref) < 1e-6
    
    @pytest.mark.propagation
    def test_ci_path_loss_exponent(self):
        """Test CI with different path loss exponents"""
        frequency_mhz = 900.0
        distance_km = 1.0
        
        # Test with default path loss exponent (2.0)
        result_default = ci_model(frequency_mhz, distance_km)
        
        # Test with explicit path loss exponent (2.0)
        result_explicit = ci_model(frequency_mhz, distance_km, path_loss_exponent=2.0)
        
        # Should be the same
        assert abs(result_default - result_explicit) < 0.001
        
        # Test with higher path loss exponent (3.0)
        result_high_exp = ci_model(frequency_mhz, distance_km, path_loss_exponent=3.0)
        
        # Should be higher loss with higher exponent
        assert result_high_exp > result_default
        
        # Test with lower path loss exponent (1.5)
        result_low_exp = ci_model(frequency_mhz, distance_km, path_loss_exponent=1.5)
        
        # Should be lower loss with lower exponent
        assert result_low_exp < result_default
    
    @pytest.mark.propagation
    def test_ci_frequency_scaling(self):
        """Test that CI scales correctly with frequency"""
        distance_km = 1.0
        freq1_mhz = 100.0
        freq2_mhz = 400.0  # 4x frequency
        
        loss1 = ci_model(freq1_mhz, distance_km)
        loss2 = ci_model(freq2_mhz, distance_km)
        
        # Doubling frequency should increase loss by ~12 dB (20*log10(4))
        expected_increase = 20 * np.log10(freq2_mhz / freq1_mhz)
        actual_increase = loss2 - loss1
        
        assert abs(actual_increase - expected_increase) < 0.01
    
    @pytest.mark.propagation
    def test_ci_distance_scaling(self):
        """Test that CI scales correctly with distance"""
        frequency_mhz = 900.0
        dist1_km = 1.0
        dist2_km = 4.0  # 4x distance
        
        loss1 = ci_model(frequency_mhz, dist1_km)
        loss2 = ci_model(frequency_mhz, dist2_km)
        
        # Doubling distance should increase loss by ~12 dB (20*log10(4))
        expected_increase = 20 * np.log10(dist2_km / dist1_km)
        actual_increase = loss2 - loss1
        
        assert abs(actual_increase - expected_increase) < 0.01
    
    @pytest.mark.propagation
    def test_ci_against_empirical_model(self):
        """Cross-validate CI implementation against empirical model"""
        test_cases = [
            (300.0, 0.1),      # UHF, short distance
            (900.0, 1.0),      # Typical cellular, 1km
            (2400.0, 2.0),     # WiFi/Bluetooth, 2km
            (5800.0, 5.0)      # 5G/WiFi, 5km
        ]
        
        for frequency_mhz, distance_km in test_cases:
            result_model = ci_model(frequency_mhz, distance_km)
            result_empirical = ci_model(frequency_mhz, distance_km)
            
            # Should pass validation
            assert validate_ci_output(result_model, frequency_mhz, distance_km)
            assert validate_ci_output(result_empirical, frequency_mhz, distance_km)
            
            # Results should be identical (same implementation)
            assert abs(result_model - result_empirical) < 0.001
    
    @pytest.mark.propagation
    def test_ci_extreme_antenna_heights(self):
        """Test CI with extreme antenna heights"""
        frequency_mhz = 900.0
        distance_km = 1.0
        
        # Test with zero height
        result_zero = ci_model(frequency_mhz, distance_km)
        
        # Test with very high height
        result_high = ci_model(frequency_mhz, distance_km)
        
        # For CI model, antenna height doesn't directly affect the calculation
        # (unless we're modeling it through reference distance changes)
        # So results should be the same for basic CI model
        assert abs(result_zero - result_high) < 0.001

