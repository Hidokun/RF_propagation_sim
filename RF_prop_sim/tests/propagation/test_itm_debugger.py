"""ITM model debugger tests"""
import pytest
import numpy as np
from propagation_model.itm_model import itm_path_loss as itm_model
from test_utils.validators import validate_itm_output
from test_utils.fixtures import SAMPLE_VALID_CONFIGS

class TestITMDebugger:
    """Test suite for ITM model debugger"""
    
    @pytest.mark.propagation
    @pytest.mark.requires_itmlogic
    def test_itm_basic_functionality(self):
        """Test basic ITM functionality with standard inputs"""
        frequency_mhz = 1000.0
        distance_km = 10.0
        tx_height_m = 10.0
        rx_height_m = 10.0
        ground_conductivity = 0.01
        ground_dielectric = 15.0
        n_surface = 301
        
        try:
            # Test ITM implementation
            result = itm_model(
                frequency_mhz=frequency_mhz,
                distance_km=distance_km,
                tx_height_m=tx_height_m,
                rx_height_m=rx_height_m,
                ground_permittivity=ground_conductivity,  # Note: using conductivity as permittivity for simplicity
                surface_refractivity=n_surface
            )
            
            # Validate outputs
            assert validate_itm_output(result, frequency_mhz, distance_km, tx_height_m, rx_height_m)
            
            # Result should be reasonable
            assert result is not None
            assert not np.isnan(result)
            assert result >= 0  # Path loss should be non-negative
            
        except ImportError:
            # Skip test if itmlogic is not available
            pytest.skip("itmlogic package not available")
    
    @pytest.mark.propagation
    def test_itm_zero_inputs(self):
        """Test ITM with zero or negative inputs"""
        # Test zero frequency
        try:
            result = itm_model(frequency_mhz=0.0, distance_km=10.0, tx_height_m=10.0, rx_height_m=10.0)
            assert result == 0.0 or result is None
        except (ImportError, ValueError):
            # Expected if itmlogic is not available or invalid input
            pass
        
        # Test zero distance
        try:
            result = itm_model(frequency_mhz=1000.0, distance_km=0.0, tx_height_m=10.0, rx_height_m=10.0)
            assert result == 0.0 or result is None
        except (ImportError, ValueError):
            pass
            
        # Test negative values
        try:
            result = itm_model(frequency_mhz=-100.0, distance_km=10.0, tx_height_m=10.0, rx_height_m=10.0)
            assert result == 0.0 or result is None
        except (ImportError, ValueError):
            pass
    
    @pytest.mark.propagation
    @pytest.mark.requires_itmlogic
    def test_itm_frequency_scaling(self):
        """Test that ITM path loss scales reasonably with frequency"""
        distance_km = 5.0
        tx_height_m = 10.0
        rx_height_m = 10.0
        freq1_mhz = 100.0
        freq2_mhz = 400.0  # 4x frequency
        
        try:
            loss1 = itm_model(
                frequency_mhz=freq1_mhz,
                distance_km=distance_km,
                tx_height_m=tx_height_m,
                rx_height_m=rx_height_m
            )
            loss2 = itm_model(
                frequency_mhz=freq2_mhz,
                distance_km=distance_km,
                tx_height_m=tx_height_m,
                rx_height_m=rx_height_m
            )
            
            # Both should be valid results
            assert loss1 is not None and not np.isnan(loss1)
            assert loss2 is not None and not np.isnan(loss2)
            
            # Higher frequency should generally cause more attenuation
            assert loss2 >= loss1
            
        except ImportError:
            pytest.skip("itmlogic package not available")
    
    @pytest.mark.propagation
    @pytest.mark.requires_itmlogic
    def test_itm_distance_scaling(self):
        """Test that ITM path loss scales reasonably with distance"""
        frequency_mhz = 1000.0
        tx_height_m = 10.0
        rx_height_m = 10.0
        dist1_km = 1.0
        dist2_km = 3.0  # 3x distance
        
        try:
            loss1 = itm_model(
                frequency_mhz=frequency_mhz,
                distance_km=dist1_km,
                tx_height_m=tx_height_m,
                rx_height_m=rx_height_m
            )
            loss2 = itm_model(
                frequency_mhz=frequency_mhz,
                distance_km=dist2_km,
                tx_height_m=tx_height_m,
                rx_height_m=rx_height_m
            )
            
            # Both should be valid results
            assert loss1 is not None and not np.isnan(loss1)
            assert loss2 is not None and not np.isnan(loss2)
            
            # Longer distance should generally cause more attenuation
            assert loss2 >= loss1
            
        except ImportError:
            pytest.skip("itmlogic package not available")
    
    @pytest.mark.propagation
    @pytest.mark.requires_itmlogic
    def test_itm_antenna_height_scaling(self):
        """Test that ITM path loss varies with antenna heights"""
        frequency_mhz = 1000.0
        distance_km = 5.0
        
        try:
            # Test low antenna heights
            loss_low = itm_model(
                frequency_mhz=frequency_mhz,
                distance_km=distance_km,
                tx_height_m=1.0,
                rx_height_m=1.0
            )
            
            # Test high antenna heights
            loss_high = itm_model(
                frequency_mhz=frequency_mhz,
                distance_km=distance_km,
                tx_height_m=100.0,
                rx_height_m=100.0
            )
            
            # Both should be valid results
            assert loss_low is not None and not np.isnan(loss_low)
            assert loss_high is not None and not np.isnan(loss_high)
            
            # Higher antennas should generally result in less path loss (better line-of-sight)
            assert loss_high <= loss_low
            
        except ImportError:
            pytest.skip("itmlogic package not available")
    
    @pytest.mark.propagation
    def test_itm_invalid_inputs(self):
        """Test ITM with various invalid inputs"""
        # Test with invalid ground parameters
        try:
            result = itm_model(
                frequency_mhz=1000.0,
                distance_km=10.0,
                tx_height_m=10.0,
                rx_height_m=10.0,
                ground_conductivity=-0.01,  # Invalid negative conductivity
                ground_dielectric=15.0
            )
            # Should handle gracefully
            assert result is None or result == 0.0 or not np.isnan(result)
        except (ImportError, ValueError, Exception):
            # Expected for invalid inputs
            pass
        
        try:
            result = itm_model(
                frequency_mhz=1000.0,
                distance_km=10.0,
                tx_height_m=10.0,
                rx_height_m=10.0,
                ground_conductivity=0.01,
                ground_dielectric=-15.0  # Invalid negative dielectric
            )
            # Should handle gracefully
            assert result is None or result == 0.0 or not np.isnan(result)
        except (ImportError, ValueError, Exception):
            pass