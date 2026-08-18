"""Error propagation integration debugger tests"""
import pytest
import json
import tempfile
import os
from RF_prop_sim.input_data_collection.ingestion import SimulationConfig
from RF_prop_sim.antenna_data.parser import AntennaConfig
from propagation_model.models import free_space_path_loss, rain_attenuation
from test_utils.fixtures import SAMPLE_VALID_CONFIGS

class TestErrorPropagationDebugger:
    """Test suite for error propagation integration debugger"""
    
    @pytest.mark.integration
    def test_config_error_propagation(self):
        """Test that configuration errors propagate appropriately"""
        # Test with invalid frequency
        invalid_config_data = {
            "frequency": -1000.0,  # Invalid negative frequency
            "distance": 5.0,
            "tx_height": 10.0,
            "rx_height": 1.5
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config_data, f)
            config_path = f.name
        
        try:
            # Should raise appropriate exception for invalid config
            with pytest.raises((ValueError, AssertionError)):
                SimulationConfig(config_path)
        finally:
            os.unlink(config_path)
    
    @pytest.mark.integration
    def test_antenna_config_error_propagation(self):
        """Test that antenna configuration errors propagate appropriately"""
        # Test with invalid frequency
        invalid_antenna_data = {
            "gain": 15.0,
            "beamwidth": 65.0,
            "polarization": "vertical",
            "frequency": -1000.0  # Invalid negative frequency
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_antenna_data, f)
            antenna_path = f.name
        
        try:
            # Should raise appropriate exception for invalid antenna config
            with pytest.raises((ValueError, AssertionError)):
                AntennaConfig(antenna_path)
        finally:
            os.unlink(antenna_path)
    
    @pytest.mark.integration
    def test_propagation_model_error_handling(self):
        """Test that propagation models handle invalid inputs gracefully"""
        # Test FSPL with invalid inputs
        result = free_space_path_loss(-1000.0, 5.0)  # Negative frequency
        # Should handle gracefully (return 0 or similar based on implementation)
        assert result is not None
        
        result = free_space_path_loss(3000.0, -5.0)  # Negative distance
        assert result is not None
        
        # Test rain attenuation with invalid inputs
        result = rain_attenuation(-10.0, 5.0, 25.0)  # Negative frequency
        assert result is not None
        
        result = rain_attenuation(10.0, -5.0, 25.0)  # Negative distance
        assert result is not None
        
        result = rain_attenuation(10.0, 5.0, -5.0)  # Negative rain rate
        assert result is not None
    
    @pytest.mark.integration
    def test_error_propagation_through_pipeline(self):
        """Test error propagation through complete pipeline"""
        # Test that early errors prevent later stages from executing
        invalid_config_data = {
            "frequency": 0.0,  # Invalid zero frequency
            "distance": 5.0,
            "tx_height": 10.0,
            "rx_height": 1.5
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as antenna_file:
            
            json.dump(invalid_config_data, config_file)
            config_path = config_file.name
            
            # Valid antenna config
            valid_antenna_data = {
                "gain": 15.0,
                "beamwidth": 65.0,
                "polarization": "vertical",
                "frequency": 3000.0
            }
            json.dump(valid_antenna_data, antenna_file)
            antenna_path = antenna_file.name
        
        try:
            # Config parsing should fail
            with pytest.raises((ValueError, AssertionError)):
                SimulationConfig(config_path)
            
            # Antenna config should still work (independent validation)
            antenna_config = AntennaConfig(antenna_path)
            assert antenna_config is not None
            
        finally:
            os.unlink(config_path)
            os.unlink(antenna_path)
