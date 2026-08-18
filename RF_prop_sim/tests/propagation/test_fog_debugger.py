"""Fog attenuation model debugger tests"""
import pytest
import numpy as np
from propagation_model.models import fog_attenuation as fog_model
from propagation_model.empirical_models import fog_attenuation as fog_empirical
from test_utils.validators import validate_fog_attenuation_output
from test_utils.fixtures import SAMPLE_VALID_CONFIGS

class TestFogDebugger:
    """Test suite for Fog attenuation model debugger"""
    
    @pytest.mark.propagation
    def test_fog_basic_functionality(self):
        """Test basic fog attenuation functionality with standard inputs"""
        frequency_ghz = 100.0
        distance_km = 2.0
        fog_density_gm3 = 0.5
        
        # Test both implementations
        result_model = fog_model(frequency_ghz, distance_km, fog_density_gm3)
        result_empirical = fog_empirical(frequency_ghz, distance_km, fog_density_gm3)
        
        # Validate outputs
        assert validate_fog_attenuation_output(result_model, frequency_ghz, distance_km, fog_density_gm3)
        assert validate_fog_attenuation_output(result_empirical, frequency_ghz, distance_km, fog_density_gm3)
        
        # Results should be reasonable
        assert result_model >= 0
        assert result_empirical >= 0
    
    @pytest.mark.propagation
    def test_fog_zero_inputs(self):
        """Test fog attenuation with zero or negative inputs"""
        # Test zero frequency
        result = fog_model(0.0, 2.0, 0.5)
        assert result == 0.0
        
        # Test zero distance
        result = fog_model(100.0, 0.0, 0.5)
        assert result == 0.0
        
        # Test zero fog density
        result = fog_model(100.0, 2.0, 0.0)
        assert result == 0.0
        
        # Test negative values
        result = fog_model(-10.0, 2.0, 0.5)
        assert result == 0.0
        
        result = fog_model(100.0, -1.0, 0.5)
        assert result == 0.0
        
        result = fog_model(100.0, 2.0, -0.1)
        assert result == 0.0
    
    @pytest.mark.propagation
    def test_fog_frequency_scaling(self):
        """Test that fog attenuation scales with frequency squared"""
        distance_km = 2.0
        fog_density_gm3 = 0.5
        freq1_ghz = 10.0
        freq2_ghz = 40.0  # 4x frequency
        
        loss1 = fog_model(freq1_ghz, distance_km, fog_density_gm3)
        loss2 = fog_model(freq2_ghz, distance_km, fog_density_gm3)
        
        # Higher frequency should cause more attenuation (frequency squared dependence)
        assert loss2 >= loss1
        
        # Should be roughly proportional to frequency squared
        expected_ratio = (freq2_ghz / freq1_ghz) ** 2
        actual_ratio = loss2 / loss1 if loss1 > 0 else float('inf')
        
        # Allow tolerance for the simplified model
        assert abs(actual_ratio - expected_ratio) < expected_ratio * 0.5
    
    @pytest.mark.propagation
    def test_fog_distance_scaling(self):
        """Test that fog attenuation scales linearly with distance"""
        frequency_ghz = 100.0
        fog_density_gm3 = 0.5
        dist1_km = 1.0
        dist2_km = 3.0  # 3x distance
        
        loss1 = fog_model(frequency_ghz, dist1_km, fog_density_gm3)
        loss2 = fog_model(frequency_ghz, dist2_km, fog_density_gm3)
        
        # Should scale linearly with distance
        expected_ratio = dist2_km / dist1_km
        actual_ratio = loss2 / loss1 if loss1 > 0 else float('inf')
        
        assert abs(actual_ratio - expected_ratio) < 0.01  # Should be very close to linear
    
    @pytest.mark.propagation
    def test_fog_density_scaling(self):
        """Test that fog attenuation scales linearly with fog density"""
        frequency_ghz = 100.0
        distance_km = 2.0
        density1_gm3 = 0.1
        density2_gm3 = 0.4  # 4x density
        
        loss1 = fog_model(frequency_ghz, distance_km, density1_gm3)
        loss2 = fog_model(frequency_ghz, distance_km, density2_gm3)
        
        # Should scale linearly with density
        expected_ratio = density2_gm3 / density1_gm3
        actual_ratio = loss2 / loss1 if loss1 > 0 else float('inf')
        
        assert abs(actual_ratio - expected_ratio) < 0.01  # Should be very close to linear
    
    @pytest.mark.propagation
    def test_fog_against_empirical_model(self):
        """Cross-validate fog attenuation implementation against empirical model"""
        test_cases = [
            (10.0, 0.5, 0.1),    # Low frequency, short distance, low density
            (100.0, 2.0, 0.5),   # Medium frequency, medium distance, medium density
            (200.0, 5.0, 2.0),   # High frequency, long distance, high density
        ]
        
        for frequency_ghz, distance_km, fog_density_gm3 in test_cases:
            result_model = fog_model(frequency_ghz, distance_km, fog_density_gm3)
            result_empirical = fog_empirical(frequency_ghz, distance_km, fog_density_gm3)
            
            # Should pass validation
            assert validate_fog_attenuation_output(result_model, frequency_ghz, distance_km, fog_density_gm3)
            assert validate_fog_attenuation_output(result_empirical, frequency_ghz, distance_km, fog_density_gm3)
            
            # Both should be reasonable values
            assert result_model >= 0
            assert result_empirical >= 0
            
            # For the simplified models, they should be reasonably close
            if result_model > 0 and result_empirical > 0:
                ratio = result_model / result_empirical
                assert 0.5 <= ratio <= 2.0  # Within factor of 2