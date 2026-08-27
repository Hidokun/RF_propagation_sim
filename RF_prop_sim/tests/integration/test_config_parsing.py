"""Config parsing integration debugger tests"""
import pytest
import json
import tempfile
import os
from RF_prop_sim.input_data_collection.ingestion import (
    SimulationConfig,
    parse_simulation_config,
    load_config_from_file,
)
from test_utils.fixtures import SAMPLE_VALID_CONFIGS, SAMPLE_INVALID_CONFIGS


def _write_json(data):
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(data, f)
    f.close()
    return f.name


class TestConfigParsingDebugger:
    """Test suite for config parsing integration debugger"""

    @pytest.mark.integration
    def test_valid_config_parsing(self):
        """Test parsing of valid configuration files"""
        for config_name, config_data in SAMPLE_VALID_CONFIGS.items():
            config_path = _write_json(config_data)
            try:
                config = parse_simulation_config(config_path)
                assert isinstance(config, SimulationConfig)
                # Canonical field names must be present
                assert hasattr(config, 'frequency_mhz')
                assert hasattr(config, 'distance_km')
                assert config.model == config_data['model']
            finally:
                os.unlink(config_path)

    @pytest.mark.integration
    def test_alias_keys_are_normalized(self):
        """Alias keys (frequency, distance) map onto canonical fields"""
        raw = {"frequency": 3000.0, "distance": 5.0, "tx_height": 10.0, "rx_height": 1.5}
        config = parse_simulation_config(raw)
        assert config.frequency_mhz == 3000.0
        assert config.distance_km == 5.0
        assert config.tx_height_m == 10.0
        assert config.rx_height_m == 1.5

    @pytest.mark.integration
    def test_invalid_config_handling(self):
        """Test handling of invalid configuration files"""
        for config_name, config_data in SAMPLE_INVALID_CONFIGS.items():
            config_path = _write_json(config_data)
            try:
                # Validation must reject negative/zero frequency or distance,
                # and negative heights.
                with pytest.raises((ValueError, AssertionError, TypeError)):
                    parse_simulation_config(config_path)
            finally:
                os.unlink(config_path)

    @pytest.mark.integration
    def test_missing_file_raises(self):
        """Nonexistent path raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            parse_simulation_config("no_such_config_file.json")

    @pytest.mark.integration
    def test_config_file_format_support(self):
        """Test support for different config file formats"""
        config_data = {
            "frequency": 3000.0,
            "distance": 5.0,
            "tx_height": 10.0,
            "rx_height": 1.5
        }

        # JSON format parses into canonical fields
        config_path = _write_json(config_data)
        try:
            config = parse_simulation_config(config_path)
            assert config is not None
            assert config.frequency_mhz == 3000.0
            assert config.distance_km == 5.0
        finally:
            os.unlink(config_path)

        # Unknown extension falls back to lenient key=value text parsing;
        # unrecognized content yields a default-initialized config (documented behavior).
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        f.write("invalid format")
        f.close()
        try:
            config = parse_simulation_config(f.name)
            assert isinstance(config, SimulationConfig)
            assert config.frequency_mhz > 0  # untouched default survived
        finally:
            os.unlink(f.name)

    @pytest.mark.integration
    def test_load_config_from_file_helper(self):
        """load_config_from_file delegates to the parser entrypoint"""
        config_path = _write_json(SAMPLE_VALID_CONFIGS['fspl'])
        try:
            config = load_config_from_file(config_path)
            assert config.frequency_mhz == SAMPLE_VALID_CONFIGS['fspl']['frequency_mhz']
        finally:
            os.unlink(config_path)
