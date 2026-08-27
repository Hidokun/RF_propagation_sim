"""Error propagation integration debugger tests"""
import pytest
import json
import tempfile
import os
from RF_prop_sim.input_data_collection.ingestion import parse_simulation_config
from RF_prop_sim.antenna_data.parser import parse_antenna_config
from propagation_model import free_space_path_loss, rain_attenuation


def _write_json(data):
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(data, f)
    f.close()
    return f.name


class TestErrorPropagationDebugger:
    """Test suite for error propagation integration debugger"""

    @pytest.mark.integration
    def test_config_error_propagation(self):
        """Invalid frequency must raise during config validation"""
        invalid_config_data = {
            "frequency": -1000.0,  # Invalid negative frequency (alias form)
            "distance": 5.0,
        }
        config_path = _write_json(invalid_config_data)
        try:
            with pytest.raises((ValueError, AssertionError)):
                parse_simulation_config(config_path)
        finally:
            os.unlink(config_path)

    @pytest.mark.integration
    def test_antenna_config_error_propagation(self):
        """Invalid antenna frequency must raise during validation"""
        invalid_antenna_data = {
            "gain": 15.0,
            "beamwidth": 65.0,
            "polarization": "vertical",
            "frequency": -1000.0,  # Invalid negative frequency
        }
        antenna_path = _write_json(invalid_antenna_data)
        try:
            with pytest.raises((ValueError, AssertionError)):
                parse_antenna_config(antenna_path)
        finally:
            os.unlink(antenna_path)

    @pytest.mark.integration
    def test_propagation_model_error_handling(self):
        """Propagation models handle invalid inputs gracefully (return 0)"""
        assert free_space_path_loss(-1000.0, 5.0) == 0.0   # Negative frequency
        assert free_space_path_loss(3000.0, -5.0) == 0.0   # Negative distance
        assert rain_attenuation(-10.0, 5.0, 25.0) == 0.0   # Negative frequency
        assert rain_attenuation(10.0, -5.0, 25.0) == 0.0   # Negative distance
        assert rain_attenuation(10.0, 5.0, -5.0) == 0.0    # Negative rain rate

    @pytest.mark.integration
    def test_error_propagation_through_pipeline(self):
        """Early validation errors prevent downstream stages from running"""
        invalid_config_data = {"frequency": 0.0}  # Invalid zero frequency
        valid_antenna_data = {
            "gain": 15.0,
            "beamwidth": 65.0,
            "polarization": "vertical",
            "frequency": 3000.0,
        }
        config_path = _write_json(invalid_config_data)
        antenna_path = _write_json(valid_antenna_data)
        try:
            # Config parsing fails hard...
            with pytest.raises((ValueError, AssertionError)):
                parse_simulation_config(config_path)

            # ...while independent antenna parsing still succeeds.
            antenna_config = parse_antenna_config(antenna_path)
            assert antenna_config is not None
            assert antenna_config.frequency_mhz == 3000.0
        finally:
            os.unlink(config_path)
            os.unlink(antenna_path)
