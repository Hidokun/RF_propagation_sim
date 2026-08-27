"""Fog attenuation model debugger tests"""
import pytest
import numpy as np
from propagation_model import fog_attenuation as fog_model
from test_utils.validators import validate_fog_attenuation_output


def fog_closed_form(frequency_ghz, distance_km, fog_density_gm3):
    """Closed-form reference for the P.840-style model under test."""
    return 0.2 * fog_density_gm3 * frequency_ghz ** 2 / (frequency_ghz ** 2 + 0.7) * distance_km

class TestFogDebugger:
    """Test suite for Fog attenuation model debugger"""
    
    @pytest.mark.propagation
    def test_fog_basic_functionality(self):
        """Test basic fog attenuation functionality with standard inputs"""
        frequency_ghz = 100.0
        distance_km = 2.0
        fog_density_gm3 = 0.5

        result = fog_model(frequency_ghz, distance_km, fog_density_gm3)

        # Validate output and match the closed-form reference
        assert validate_fog_attenuation_output(result, frequency_ghz, distance_km, fog_density_gm3)
        assert result >= 0
        assert abs(result - fog_closed_form(frequency_ghz, distance_km, fog_density_gm3)) < 1e-9
    
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
        """Test fog attenuation frequency dependence (ITU-R P.840-style).

        The model gamma ∝ f²/(f²+0.7) exhibits two physical regimes:
        - Rayleigh regime (f² << 0.7 GHz², f << ~0.84 GHz): grows ~ f²
        - Saturation regime (f >> ~0.84 GHz): asymptotes to 0.2*M*d
        """
        distance_km = 2.0
        fog_density_gm3 = 0.5

        # Monotonic increase across the whole band (0.01 GHz .. 1000 GHz)
        freqs = np.logspace(-2, 3, 50)
        losses = [fog_model(f, distance_km, fog_density_gm3) for f in freqs]
        assert all(b >= a for a, b in zip(losses, losses[1:]))

        # Rayleigh regime: 2x frequency -> ratio close to 4 (with the
        # known denominator correction factor from the closed form)
        f1, f2 = 0.05, 0.10
        loss1 = fog_model(f1, distance_km, fog_density_gm3)
        loss2 = fog_model(f2, distance_km, fog_density_gm3)
        actual_ratio = loss2 / loss1
        expected_ratio = fog_closed_form(f2, distance_km, fog_density_gm3) / \
            fog_closed_form(f1, distance_km, fog_density_gm3)
        assert abs(actual_ratio - expected_ratio) < 1e-9
        assert 3.5 < actual_ratio < 4.0  # near-quadratic growth

        # Saturation regime: 10 -> 40 GHz stays nearly flat (ratio < 1.1),
        # NOT quadratic — this is correct P.840 behavior above ~10 GHz.
        loss_hi1 = fog_model(10.0, distance_km, fog_density_gm3)
        loss_hi2 = fog_model(40.0, distance_km, fog_density_gm3)
        saturation_ratio = loss_hi2 / loss_hi1
        assert 1.0 <= saturation_ratio < 1.1
    
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
    def test_fog_against_closed_form(self):
        """Validate implementation against the closed-form expression"""
        test_cases = [
            (10.0, 0.5, 0.1),    # Low frequency, short distance, low density
            (100.0, 2.0, 0.5),   # Medium frequency, medium distance, medium density
            (200.0, 5.0, 2.0),   # High frequency, long distance, high density
        ]

        for frequency_ghz, distance_km, fog_density_gm3 in test_cases:
            result = fog_model(frequency_ghz, distance_km, fog_density_gm3)

            assert validate_fog_attenuation_output(result, frequency_ghz, distance_km, fog_density_gm3)
            assert result >= 0

            expected = fog_closed_form(frequency_ghz, distance_km, fog_density_gm3)
            assert abs(result - expected) < 1e-9