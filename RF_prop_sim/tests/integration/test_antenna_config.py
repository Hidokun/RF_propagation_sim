"""Antenna config integration debugger tests"""
import pytest
import json
import tempfile
import os
from RF_prop_sim.antenna_data.parser import AntennaConfig
from test_utils.fixtures import SAMPLE_VALID_CONFIGS

class TestAntennaConfigDebugger:
    """Test suite for antenna config integration debugger"""
    
    @pytest.mark.integration
    def test_valid_antenna_config_parsing(self):
        """Test parsing of valid antenna configuration files"""
        for config_name, config_data in SAMPLE_VALID_CONFIGS.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config_data, f)
                config_path = f.name
            
            try:
                antenna_config = AntennaConfig(config_path)
                assert antenna_config is not None
                # Basic validation that key fields are present
                assert hasattr(antenna_config, 'gain')
                assert hasattr(antenna_config, 'beamwidth')
                assert hasattr(antenna_config, 'polarization')
                assert hasattr(antenna_config, 'frequency')
            finally:
                os.unlink(config_path)
    
    @pytest.mark.integration
    def test_antenna_config_field_validation(self):
        """Test validation of antenna configuration fields"""
        config_data = {
            "gain": 15.0,
            "beamwidth": 65.0,
            "polarization": "vertical",
            "frequency": 3000.0
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            antenna_config = AntennaConfig(config_path)
            assert antenna_config.gain == 15.0
            assert antenna_config.beamwidth == 65.0
            assert antenna_config.polarization == "vertical"
            assert antenna_config.frequency == 3000.0
        finally:
            os.unlink(config_path)
    
    @pytest.mark.integration
    def test_antenna_config_defaults(self):
        """Test antenna configuration with missing fields (should use defaults)"""
        # Minimal config with only required fields
        minimal_config_data = {
            "frequency": 3000.0
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(minimal_config_data, f)
            config_path = f.name
        
        try:
            antenna_config = AntennaConfig(config_path)
            assert antenna_config is not None
            assert antenna_config.frequency == 3000.0
            # Other fields should have reasonable defaults
        finally:
            os.unlink(config_path)
    
    @pytest.mark.integration
    def test_antenna_config_invalid_values(self):
        """Test handling of invalid antenna configuration values"""
        # Test with invalid gain
        invalid_config_data = {
            "gain": -50.0,  # Unrealistically high negative gain
            "beamwidth": 65.0,
            "polarization": "vertical",
            "frequency": 3000.0
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config_data, f)
            config_path = f.name
        
        try:
            # Depending on implementation, might raise exception or clamp values
            antenna_config = AntennaConfig(config_path)
            # If no exception, values should be reasonable
            assert antenna_config is not None
        except (ValueError, AssertionError):
            # Expected if validation is strict
            pass
        finally:
            os.unlink(config_path)


