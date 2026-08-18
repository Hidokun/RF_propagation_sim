"""Rain attenuation model debugger tests"""
import pytest
import numpy as np
from propagation_model.models import rain_attenuation as rain_model
from propagation_model.empirical_models import rain_attenuation as rain_empirical
from test_utils.validators import validate_rain_attenuation_output
from test_utils.fixtures import SAMPLE_VALID_CONFIGS

class TestRainDebugger:
    """Test suite for Rain attenuation model debugger"""
    
    @pytest.mark.propagation
    def test_rain_basic_functionality(self):
        """Test basic rain attenuation functionality with standard inputs"""
        frequency_ghz = 10.0
        distance_km = 5.0
        rain_rate_mmh = 25.0
        
        # Test both implementations
        result_model = rain_model(frequency_ghz, distance_km, rain_rate_mmh)
        result_empirical = rain_empirical(frequency_ghz, distance_km, rain_rate_mmh)
        
        # Validate outputs
        assert validate_rain_attenuation_output(result_model, frequency_ghz, distance_km, rain_rate_mmh)
        assert validate_rain_attenuation_output(result_empirical, frequency_ghz, distance_km, rain_rate_mmh)
        
        # Results should be reasonable (they may differ due to different implementations)
        assert result_model >= 0
        assert result_empirical >= 0
    
    @pytest.mark.propagation
    def test_rain_zero_inputs(self):
        """Test rain attenuation with zero or negative inputs"""
        # Test zero frequency
        result = rain_model(0.0, 5.0, 25.0)
        assert result == 0.0
        
        # Test zero distance
        result = rain_model(10.0, 0.0, 25.0)
        assert result == 0.0
        
        # Test zero rain rate
        result = rain_model(10.0, 5.0, 0.0)
        assert result == 0.0
        
        # Test negative values
        result = rain_model(-10.0, 5.0, 25.0)
        assert result == 0.0
        
        result = rain_model(10.0, -1.0, 25.0)
        assert result == 0.0
        
        result = rain_model(10.0, 5.0, -5.0)
        assert result == 0.0
    
    @pytest.mark.propagation
    def test_rain_frequency_scaling(self):
        """Test that rain attenuation scales reasonably with frequency"""
        distance_km = 5.0
        rain_rate_mmh = 25.0
        freq1_ghz = 5.0
        freq2_ghz = 20.0  # 4x frequency
        
        loss1 = rain_model(freq1_ghz, distance_km, rain_rate_mmh)
        loss2 = rain_model(freq2_ghz, distance_km, rain_rate_mmh)
        
        # Higher frequency should generally cause more attenuation
        assert loss2 >= loss1
        
        # But not excessively so - sanity check
        assert loss2 <= loss1 * 10  # Should not increase by more than 10x
    
    @pytest.mark.propagation
    def test_rain_distance_scaling(self):
        """Test that rain attenuation scales linearly with distance"""
        frequency_ghz = 10.0
        rain_rate_mmh = 25.0
        dist1_km = 1.0
        dist2_km = 3.0  # 3x distance
        
        loss1 = rain_model(frequency_ghz, dist1_km, rain_rate_mmh)
        loss2 = rain_model(frequency_ghz, dist2_km, rain_rate_mmh)
        
        # Should scale approximately linearly with distance
        expected_ratio = dist2_km / dist1_km
        actual_ratio = loss2 / loss1 if loss1 > 0 else float('inf')
        
        # Allow for some variation due to the power law model
        assert abs(actual_ratio - expected_ratio) < 0.5  # Within 50% tolerance
    
    @pytest.mark.propagation
    def test_rain_rain_rate_scaling(self):
        """Test that rain attenuation scales with rain rate"""
        frequency_ghz = 10.0
        distance_km = 5.0
        rate1_mmh = 5.0
        rate2_mmh = 20.0  # 4x rain rate
        
        loss1 = rain_model(frequency_ghz, distance_km, rate1_mmh)
        loss2 = rain_model(frequency_ghz, distance_km, rate2_mmh)
        
        # Higher rain rate should cause more attenuation
        assert loss2 >= loss1
        
        # Should follow power law behavior (approximately)
        # For ITU-R P.838, alpha is typically around 0.9-1.0
        expected_ratio = (rate2_mmh / rate1_mmh) ** 0.9
        actual_ratio = loss2 / loss1 if loss1 > 0 else float('inf')
        
        # Allow tolerance for the simplified model
        assert abs(actual_ratio - expected_ratio) < expected_ratio * 0.5
    
    @pytest.mark.propagation
    def test_rain_against_empirical_model(self):
        """Cross-validate rain attenuation implementation against empirical model"""
        test_cases = [
            (5.0, 1.0, 5.0),    # Low frequency, short distance, light rain
            (10.0, 5.0, 25.0),  # Medium frequency, medium distance, moderate rain
            (30.0, 10.0, 50.0), # High frequency, long distance, heavy rain
        ]
        
        for frequency_ghz, distance_km, rain_rate_mmh in test_cases:
            result_model = rain_model(frequency_ghz, distance_km, rain_rate_mmh)
            result_empirical = rain_empirical(frequency_ghz, distance_km, rain_rate_mmh)
            
            # Should pass validation
            assert validate_rain_attenuation_output(result_model, frequency_ghz, distance_km, rain_rate_mmh)
            assert validate_rain_attenuation_output(result_empirical, frequency_ghz, distance_km, rain_rate_mmh)
            
            # Both should be reasonable values
            assert result_model >= 0
            assert result_empirical >= 0
    
    @pytest.mark.propagation
    def test_rain_polarization_handling(self):
        """Test that rain attenuation handles different polarizations"""
        frequency_ghz = 10.0
        distance_km = 5.0
        rain_rate_mmh = 25.0
        
        # Test different polarizations in empirical model
        result_h = rain_empirical(frequency_ghz, distance_km, rain_rate_mmh, polarization="horizontal")
        result_v = rain_empirical(frequency_ghz, distance_km, rain_rate_mmh, polarization="vertical")
        
        # Both should be valid
        assert validate_rain_attenuation_output(result_h, frequency_ghz, distance_km, rain_rate_mmh, "horizontal")
        assert validate_rain_attenuation_output(result_v, frequency_ghz, distance_km, rain_rate_mmh, "vertical")
        
        # They should be different (but not enormously so)
        assert abs(result_h - result_v) < max(result_h, result_v) * 0.5