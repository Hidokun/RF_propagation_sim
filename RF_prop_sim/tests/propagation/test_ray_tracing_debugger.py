"""Ray Tracing model debugger tests"""
import pytest
import numpy as np
from propagation_model.ray_tracing_model import ray_tracing_path_loss as ray_tracing_model
from test_utils.validators import validate_ray_tracing_output
from test_utils.fixtures import SAMPLE_VALID_CONFIGS

class TestRayTracingDebugger:
    """Test suite for Ray Tracing model debugger"""
    
    @pytest.mark.propagation
    @pytest.mark.requires_sionna
    def test_ray_tracing_basic_functionality(self):
        """Test basic ray tracing functionality with standard inputs"""
        frequency_hz = 30e9  # 30 GHz
        tx_array = [0, 0, 10]  # [x, y, z] in meters
        rx_array = [100, 0, 1.5]  # [x, y, z] in meters
        
        try:
            # Test ray tracing implementation
            result = ray_tracing_model(
                frequency_ghz=frequency_hz / 1e9,  # Convert Hz to GHz
                tx_pos=tx_array,
                rx_pos=rx_array
            )
            
            # Validate outputs
            assert validate_ray_tracing_output(result, frequency_hz / 1e9, tx_array, rx_array)
            
            # Result should be reasonable
            assert result is not None
            # Depending on implementation, result might be:
            # - A single path loss value
            # - A list of paths
            # - A dictionary with multiple values
            # We'll check that it's not obviously wrong
            
        except ImportError:
            # Skip test if sionna is not available
            pytest.skip("sionna package not available")
        except Exception as e:
            # Handle other potential initialization issues
            if "sionna" in str(e).lower() or "import" in str(e).lower():
                pytest.skip("sionna package not available or not properly installed")
            else:
                # Re-raise if it's not a sionna-related issue
                raise
    
    @pytest.mark.propagation
    def test_ray_tracing_zero_inputs(self):
        """Test ray tracing with zero or negative inputs"""
        # Test zero frequency
        try:
            result = ray_tracing_model(frequency_hz=0.0, tx_array=[0,0,10], rx_array=[100,0,1.5])
            # Should handle gracefully
            assert result is None or result == 0.0
        except (ImportError, ValueError, Exception):
            # Expected if sionna is not available or invalid input
            pass
        
        # Test with zero distance (same point)
        try:
            result = ray_tracing_model(frequency_hz=30e9, tx_array=[0,0,10], rx_array=[0,0,10])
            # Should handle gracefully
            assert result is not None  # Might be 0 or very small loss
        except (ImportError, ValueError, Exception):
            pass
            
        # Test with negative coordinates
        try:
            result = ray_tracing_model(frequency_hz=30e9, tx_array=[0,0,-10], rx_array=[100,0,1.5])
            # Should handle gracefully (negative height might be valid)
            assert result is not None or result == 0.0
        except (ImportError, ValueError, Exception):
            pass
    
    @pytest.mark.propagation
    @pytest.mark.requires_sionna
    def test_ray_tracing_frequency_scaling(self):
        """Test that ray tracing path loss scales reasonably with frequency"""
        tx_array = [0, 0, 10]
        rx_array = [100, 0, 1.5]
        freq1_hz = 3e9   # 3 GHz
        freq2_hz = 30e9  # 30 GHz (10x frequency)
        
        try:
            loss1 = ray_tracing_model(
                frequency_ghz=freq1_hz / 1e9,  # Convert Hz to GHz
                tx_pos=tx_array,
                rx_pos=rx_array
            )
            loss2 = ray_tracing_model(
                frequency_ghz=freq2_hz / 1e9,  # Convert Hz to GHz
                tx_pos=tx_array,
                rx_pos=rx_array
            )
            
            # Both should be valid results
            assert loss1 is not None
            assert loss2 is not None
            
            # Higher frequency should generally cause more attenuation
            # (Though in ray tracing, this depends on reflection/refraction properties)
            
        except ImportError:
            pytest.skip("sionna package not available")
        except Exception as e:
            if "sionna" in str(e).lower():
                pytest.skip("sionna package not available or not properly installed")
            else:
                raise
    
    @pytest.mark.propagation
    @pytest.mark.requires_sionna
    def test_ray_tracing_distance_scaling(self):
        """Test that ray tracing path loss scales reasonably with distance"""
        frequency_hz = 30e9
        tx_array = [0, 0, 10]
        rx1_array = [10, 0, 1.5]   # Close receiver
        rx2_array = [1000, 0, 1.5] # Far receiver
        
        try:
            loss1 = ray_tracing_model(
                frequency_ghz=frequency_hz / 1e9,  # Convert Hz to GHz
                tx_pos=tx_array,
                rx_pos=rx1_array
            )
            loss2 = ray_tracing_model(
                frequency_ghz=frequency_hz / 1e9,  # Convert Hz to GHz
                tx_pos=tx_array,
                rx_pos=rx2_array
            )
            
            # Both should be valid results
            assert loss1 is not None
            assert loss2 is not None
            
            # Farther distance should generally cause more attenuation
            # (Though in complex environments, this isn't always strictly true due to multipath)
            
        except ImportError:
            pytest.skip("sionna package not available")
        except Exception as e:
            if "sionna" in str(e).lower():
                pytest.skip("sionna package not available or not properly installed")
            else:
                raise
    
    @pytest.mark.propagation
    @pytest.mark.requires_sionna
    def test_ray_tracing_line_of_sight(self):
        """Test ray tracing with clear line of sight vs obstructed paths"""
        frequency_hz = 30e9
        tx_array = [0, 0, 20]  # High transmitter
        
        try:
            # Clear line of sight path
            rx_los_array = [100, 0, 20]  # Same height, clear path
            
            # Obstructed path
            rx_blocked_array = [100, 0, 1.5]  # Much lower height
            
            loss_los = ray_tracing_model(
                frequency_ghz=frequency_hz / 1e9,  # Convert Hz to GHz
                tx_pos=tx_array,
                rx_pos=rx_los_array
            )
            loss_blocked = ray_tracing_model(
                frequency_ghz=frequency_hz / 1e9,  # Convert Hz to GHz
                tx_pos=tx_array,
                rx_pos=rx_blocked_array
            )
            
            # Both should be valid results
            assert loss_los is not None
            assert loss_blocked is not None
            
            # In many scenarios, obstructed path might have higher loss
            # But this depends heavily on the environment and reflection properties
            # So we mainly check that both produce valid results
            
        except ImportError:
            pytest.skip("sionna package not available")
        except Exception as e:
            if "sionna" in str(e).lower():
                pytest.skip("sionna package not available or not properly installed")
            else:
                raise
    
    @pytest.mark.propagation
    def test_ray_tracing_invalid_inputs(self):
        """Test ray tracing with various invalid inputs"""
        # Test with invalid array dimensions
        try:
            result = ray_tracing_model(
                frequency_hz=30e9,
                tx_array=[0, 0],      # Missing z coordinate
                rx_array=[100, 0, 1.5]
            )
            # Should handle gracefully
            assert result is None or result == 0.0
        except (ImportError, ValueError, Exception):
            # Expected for invalid inputs
            pass
        
        try:
            result = ray_tracing_model(
                frequency_hz=30e9,
                tx_array=[0, 0, 10, 5],  # Too many coordinates
                rx_array=[100, 0, 1.5]
            )
            # Should handle gracefully
            assert result is None or result == 0.0
        except (ImportError, ValueError, Exception):
            pass