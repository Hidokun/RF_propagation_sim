"""Antenna config integration debugger tests"""
import pytest
import json
import tempfile
import os
from RF_prop_sim.antenna_data.parser import AntennaConfig, parse_antenna_config


def _write_json(data):
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(data, f)
    f.close()
    return f.name


class TestAntennaConfigDebugger:
    """Test suite for antenna config integration debugger"""

    @pytest.mark.integration
    def test_valid_antenna_config_parsing(self):
        """Test parsing of valid antenna configuration files"""
        config_data = {
            "frequency_mhz": 3000.0,
            "gain_dbi": 15.0,
            "beamwidth_h": 65.0,
            "polarization": "vertical",
        }
        config_path = _write_json(config_data)
        try:
            antenna = parse_antenna_config(config_path)
            assert isinstance(antenna, AntennaConfig)
            # Key fields must exist and round-trip correctly
            assert antenna.frequency_mhz == 3000.0
            assert antenna.gain_dbi == 15.0
            assert antenna.beamwidth_h == 65.0
            assert antenna.polarization == "vertical"
        finally:
            os.unlink(config_path)

    @pytest.mark.integration
    def test_antenna_config_field_aliases(self):
        """Alias keys (gain, beamwidth, frequency) map to canonical fields"""
        config_data = {
            "gain": 15.0,
            "beamwidth": 65.0,
            "polarization": "vertical",
            "frequency": 3000.0,
        }
        antenna = parse_antenna_config(config_data)
        assert antenna.gain_dbi == 15.0
        assert antenna.beamwidth_h == 65.0
        assert antenna.polarization == "vertical"
        assert antenna.frequency_mhz == 3000.0

    @pytest.mark.integration
    def test_antenna_config_defaults(self):
        """Minimal config uses sensible defaults for missing fields"""
        minimal = {"frequency": 3000.0}
        config_path = _write_json(minimal)
        try:
            antenna = parse_antenna_config(config_path)
            assert antenna.frequency_mhz == 3000.0
            # Defaults for everything else
            assert antenna.tx_power_dbm > 0
            assert antenna.height_m >= 0
            assert antenna.beamwidth_h > 0
            assert antenna.polarization in ("vertical", "horizontal", "dual")
        finally:
            os.unlink(config_path)

    @pytest.mark.integration
    def test_antenna_config_invalid_values(self):
        """Out-of-range gain warns but proceeds; invalid frequency raises"""
        # Negative gain: warned about, construction still succeeds
        warn_data = {"gain": -50.0, "frequency": 3000.0}
        antenna = parse_antenna_config(warn_data)
        assert antenna is not None
        assert antenna.gain_dbi == -50.0

        # Non-positive frequency: hard validation error
        bad_data = {"frequency": -1000.0}
        with pytest.raises(ValueError):
            parse_antenna_config(bad_data)

    @pytest.mark.integration
    def test_antenna_config_missing_file(self):
        """Missing file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            parse_antenna_config("no_such_antenna_file.json")
