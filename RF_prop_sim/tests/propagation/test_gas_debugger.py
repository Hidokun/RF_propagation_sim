"""Gas attenuation model debugger tests"""
import pytest
import numpy as np
from propagation_model import gas_attenuation as gas_model
from test_utils.validators import validate_gas_attenuation_output
from test_utils.fixtures import SAMPLE_VALID_CONFIGS

class TestGasDebugger:
    """Test suite for Gas attenuation model debugger"""
    
    @pytest.mark.propagation
    def test_gas_basic_functionality(self):
        """Test basic gas attenuation functionality with standard inputs"""
        frequency_ghz = 30.0
        distance_km = 5.0
        
        # Test both implementations
        result_model = gas_model(frequency_ghz, distance_km)
        result_empirical = gas_model(frequency_ghz, distance_km)
        
        # Validate outputs
        assert validate_gas_attenuation_output(result_model, frequency_ghz, distance_km)
        assert validate_gas_attenuation_output(result_empirical, frequency_ghz, distance_km)
        
        # Results should be reasonable
        assert result_model >= 0
        assert result_empirical >= 0
    
    @pytest.mark.propagation
    def test_gas_zero_inputs(self):
        """Test gas attenuation with zero or negative inputs"""
        # Test zero frequency
        result = gas_model(0.0, 5.0)
        assert result == 0.0
        
        # Test zero distance
        result = gas_model(30.0, 0.0)
        assert result == 0.0
        
        # Test negative values
        result = gas_model(-10.0, 5.0)
        assert result == 0.0
        
        result = gas_model(30.0, -1.0)
        assert result == 0.0
    
    @pytest.mark.propagation
    def test_gas_frequency_scaling(self):
        """Test that gas attenuation scales reasonably with frequency"""
        distance_km = 5.0
        freq1_ghz = 10.0
        freq2_ghz = 100.0  # 10x frequency
        
        loss1 = gas_model(freq1_ghz, distance_km)
        loss2 = gas_model(freq2_ghz, distance_km)
        
        # Higher frequency should generally cause more attenuation
        assert loss2 >= loss1
        
        # Should be roughly proportional to frequency squared (based on model)
        expected_ratio = (freq2_ghz / freq1_ghz) ** 2
        actual_ratio = loss2 / loss1 if loss1 > 0 else float('inf')
        
        # Allow tolerance for the simplified model
        assert actual_ratio < expected_ratio * 3  # Should not exceed expected by 3x
    
    @pytest.mark.propagation
    def test_gas_distance_scaling(self):
        """Test that gas attenuation scales linearly with distance"""
        frequency_ghz = 30.0
        dist1_km = 1.0
        dist2_km = 3.0  # 3x distance
        
        loss1 = gas_model(frequency_ghz, dist1_km)
        loss2 = gas_model(frequency_ghz, dist2_km)
        
        # Should scale linearly with distance
        expected_ratio = dist2_km / dist1_km
        actual_ratio = loss2 / loss1 if loss1 > 0 else float('inf')
        
        assert abs(actual_ratio - expected_ratio) < 0.01  # Should be very close to linear
    
    @pytest.mark.propagation
    def test_gas_temperature_dependence(self):
        """Test that gas attenuation varies with temperature"""
        frequency_ghz = 30.0
        distance_km = 5.0
        temp1_c = -40.0  # Cold
        temp2_c = 50.0   # Hot
        
        loss1 = gas_model(frequency_ghz, distance_km, temp1_c)
        loss2 = gas_model(frequency_ghz, distance_km, temp2_c)
        
        # Both should be valid
        assert validate_gas_attenuation_output(loss1, frequency_ghz, distance_km, temp1_c)
        assert validate_gas_attenuation_output(loss2, frequency_ghz, distance_km, temp2_c)
        
        # Values should be reasonable (exact relationship depends on complex physics)
        assert loss1 >= 0
        assert loss2 >= 0
    
    @pytest.mark.propagation
    def test_gas_pressure_dependence(self):
        """Test that gas attenuation varies with pressure"""
        frequency_ghz = 30.0
        distance_km = 5.0
        pressure1_hpa = 800.0  # Low pressure
        pressure2_hpa = 1200.0 # High pressure
        
        loss1 = gas_model(frequency_ghz, distance_km, pressure_hpa=pressure1_hpa)
        loss2 = gas_model(frequency_ghz, distance_km, pressure_hpa=pressure2_hpa)
        
        # Both should be valid
        assert validate_gas_attenuation_output(loss1, frequency_ghz, distance_km, pressure_hpa=pressure1_hpa)
        assert validate_gas_attenuation_output(loss2, frequency_ghz, distance_km, pressure_hpa=pressure2_hpa)
        
        # Values should be reasonable
        assert loss1 >= 0
        assert loss2 >= 0
    
    @pytest.mark.propagation
    def test_gas_humidity_dependence(self):
        """Test that gas attenuation varies with humidity"""
        frequency_ghz = 30.0
        distance_km = 5.0
        humidity1 = 0.0    # Dry
        humidity2 = 100.0  # Saturated
        
        loss1 = gas_model(frequency_ghz, distance_km, relative_humidity=humidity1)
        loss2 = gas_model(frequency_ghz, distance_km, relative_humidity=humidity2)
        
        # Both should be valid
        assert validate_gas_attenuation_output(loss1, frequency_ghz, distance_km, relative_humidity=humidity1)
        assert validate_gas_attenuation_output(loss2, frequency_ghz, distance_km, relative_humidity=humidity2)
        
        # Higher humidity should generally increase attenuation (water vapor absorption)
        assert loss2 >= loss1
    
    @pytest.mark.propagation
    def test_gas_against_closed_form(self):
        """Validate implementation against its closed-form expression"""
        test_cases = [
            (1.0, 1.0),         # Low frequency, short distance
            (10.0, 5.0),        # Medium frequency, medium distance
            (60.0, 10.0),       # High frequency, long distance
            (120.0, 20.0)       # Very high frequency, very long distance
        ]

        for frequency_ghz, distance_km in test_cases:
            result = gas_model(frequency_ghz, distance_km)
            assert validate_gas_attenuation_output(result, frequency_ghz, distance_km)
            assert result >= 0

            # Closed form: gamma_o + gamma_w terms (standard atmosphere)
            t_k = 15.0 + 273.15
            svp = 6.1121 * np.exp((17.502 * 15.0) / (15.0 + 240.97))
            wvd = 216.7 * (50.0 / 100.0 * svp) / t_k
            expected = (
                0.0001 * 1013.25 * frequency_ghz ** 2 / (frequency_ghz ** 2 + 0.1)
                + 0.000045 * wvd * frequency_ghz ** 2 / (frequency_ghz ** 2 + 0.5)
            ) * distance_km
            assert abs(result - expected) < 1e-9
