"""FSPL model debugger tests"""
import pytest
import numpy as np
from propagation_model.models import free_space_path_loss as fspl_model
from propagation_model.empirical_models import free_space_path_loss as fspl_empirical
from test_utils.validators import validate_fspl_output
from test_utils.fixtures import SAMPLE_VALID_CONFIGS

class TestFSPLDebugger:
    """Test suite for Free Space Path Loss model debugger"""
    
    @pytest.mark.propagation
    def test_fspl_basic_functionality(self):
        """Test basic FSPL functionality with standard inputs"""
        frequency_mhz = 300.0
        distance_km = 5.0
        
        # Test both implementations
        result_model = fspl_model(frequency_mhz, distance_km)
        result_empirical = fspl_empirical(frequency_mhz, distance_km)
        
        # Validate outputs
        assert validate_fspl_output(result_model, frequency_mhz, distance_km)
        assert validate_fspl_output(result_empirical, frequency_mhz, distance_km)
        
        # Results should be identical (or very close)
        assert abs(result_model - result_empirical) < 0.01
    
    @pytest.mark.propagation
    def test_fspl_zero_inputs(self):
        """Test FSPL with zero or negative inputs"""
        # Test zero frequency
        result = fspl_model(0.0, 5.0)
        assert result == 0.0
        
        # Test zero distance
        result = fspl_model(300.0, 0.0)
        assert result == 0.0
        
        # Test negative values
        result = fspl_model(-100.0, 5.0)
        assert result == 0.0
        
        result = fspl_model(300.0, -1.0)
        assert result == 0.0
    
    @pytest.mark.propagation
    def test_fspl_frequency_scaling(self):
        """Test that FSPL scales correctly with frequency"""
        distance_km = 10.0
        freq1_mhz = 100.0
        freq2_mhz = 400.0  # 4x frequency
        
        loss1 = fspl_model(freq1_mhz, distance_km)
        loss2 = fspl_model(freq2_mhz, distance_km)
        
        # Doubling frequency should increase loss by ~12 dB (20*log10(4))
        expected_increase = 20 * np.log10(freq2_mhz / freq1_mhz)
        actual_increase = loss2 - loss1
        
        assert abs(actual_increase - expected_increase) < 0.01
    
    @pytest.mark.propagation
    def test_fspl_distance_scaling(self):
        """Test that FSPL scales correctly with distance"""
        frequency_mhz = 300.0
        dist1_km = 1.0
        dist2_km = 4.0  # 4x distance
        
        loss1 = fspl_model(frequency_mhz, dist1_km)
        loss2 = fspl_model(frequency_mhz, dist2_km)
        
        # Doubling distance should increase loss by ~12 dB (20*log10(4))
        expected_increase = 20 * np.log10(dist2_km / dist1_km)
        actual_increase = loss2 - loss1
        
        assert abs(actual_increase - expected_increase) < 0.01
    
    @pytest.mark.propagation
    def test_fspl_against_empirical_model(self):
        """Cross-validate FSPL implementation against empirical model"""
        test_cases = [
            (30.0, 0.1),      # VHF, short distance
            (300.0, 1.0),     # UHF, medium distance
            (3000.0, 10.0),   # Microwave, long distance
            (30000.0, 100.0)  # mmWave, very long distance
        ]
        
        for frequency_mhz, distance_km in test_cases:
            result_model = fspl_model(frequency_mhz, distance_km)
            result_empirical = fspl_empirical(frequency_mhz, distance_km)
            
            # Should be identical
            assert abs(result_model - result_empirical) < 0.01
            
            # Should pass validation
            assert validate_fspl_output(result_model, frequency_mhz, distance_km)
            assert validate_fspl_output(result_empirical, frequency_mhz, distance_km)
    
    @pytest.mark.propagation
    def test_fspl_extreme_values(self):
        """Test FSPL with extreme frequency and distance values"""
        extreme_cases = [
            (1.0, 0.001),     # Very low frequency, very short distance
            (100000.0, 0.001), # Very high frequency, very short distance
            (1.0, 10000.0),   # Very low frequency, very long distance
            (100000.0, 10000.0) # Very high frequency, very long distance
        ]
        
        for frequency_mhz, distance_km in extreme_cases:
            result = fspl_model(frequency_mhz, distance_km)
            
            # Should not crash and should return reasonable values
            assert result is not None
            assert not np.isnan(result)
            # For extreme values, we mainly want to ensure no crashes
            # The actual values will be validated by the scaling tests above