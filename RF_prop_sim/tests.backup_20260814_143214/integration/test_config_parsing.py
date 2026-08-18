"""Config parsing integration debugger tests"""
import pytest
import json
import tempfile
import os
from RF_prop_sim.input_data_collection.ingestion import SimulationConfig
from test_utils.fixtures import SAMPLE_VALID_CONFIGS, SAMPLE_INVALID_CONFIGS

class TestConfigParsingDebugger:
    """Test suite for config parsing integration debugger"""
    
    @pytest.mark.integration
    def test_valid_config_parsing(self):
        """Test parsing of valid configuration files"""
        for config_name, config_data in SAMPLE_VALID_CONFIGS.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config_data, f)
                config_path = f.name
            
            try:
                config = SimulationConfig(config_path)
                assert config is not None
                # Basic validation that key fields are present
                assert hasattr(config, 'frequency')
                assert hasattr(config, 'distance')
            finally:
                os.unlink(config_path)
    
    @pytest.mark.integration
    def test_invalid_config_handling(self):
        """Test handling of invalid configuration files"""
        for config_name, config_data in SAMPLE_INVALID_CONFIGS.items():
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(config_data, f)
                config_path = f.name
            
            try:
                # Should handle invalid config gracefully
                with pytest.raises((ValueError, AssertionError, TypeError)):
                    SimulationConfig(config_path)
            finally:
                os.unlink(config_path)
    
    @pytest.mark.integration
    def test_config_file_format_support(self):
        """Test support for different config file formats"""
        config_data = {
            "frequency": 3000.0,
            "distance": 5.0,
            "tx_height": 10.0,
            "rx_height": 1.5
        }
        
        # Test JSON format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            config = SimulationConfig(config_path)
            assert config is not None
            assert config.frequency == 3000.0
        finally:
            os.unlink(config_path)
        
        # Test that unsupported formats raise appropriate errors
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("invalid format")
            config_path = f.name
        
        try:
            with pytest.raises(Exception):
                SimulationConfig(config_path)
        finally:
            os.unlink(config_path)