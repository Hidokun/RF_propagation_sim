"""End-to-end flow integration debugger tests"""
import pytest
import json
import tempfile
import os
from unittest.mock import patch, MagicMock

from RF_prop_sim.input_data_collection.ingestion import parse_simulation_config
from RF_prop_sim.antenna_data.parser import parse_antenna_config
from RF_prop_sim.mapping.location_service import Geocoder
from RF_prop_sim.main import run_simulation, build_antenna_config
from propagation_model import free_space_path_loss, rain_attenuation


def _write_json(data):
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(data, f)
    f.close()
    return f.name


class TestEndToEndFlowDebugger:
    """Test suite for end-to-end flow integration debugger"""

    @pytest.mark.integration
    def test_complete_simulation_pipeline(self):
        """Test complete pipeline: config -> antenna -> models -> link budget"""
        config_data = {
            "frequency": 3000.0,  # MHz (alias form)
            "distance": 5.0,      # km
            "tx_height": 10.0,    # m
            "rx_height": 1.5,     # m
        }
        antenna_data = {
            "gain": 15.0,         # dBi
            "beamwidth": 65.0,    # degrees
            "polarization": "vertical",
            "frequency": 3000.0,  # MHz
        }

        config_path = _write_json(config_data)
        antenna_path = _write_json(antenna_data)
        try:
            # Step 1-2: Parse both configs through the real parsers
            cfg = parse_simulation_config(config_path)
            assert cfg.frequency_mhz == 3000.0 and cfg.distance_km == 5.0
            antenna_cfg = parse_antenna_config(antenna_path)
            assert antenna_cfg.gain_dbi == 15.0

            # Step 3: Propagation with canonical units
            frequency_ghz = cfg.frequency_mhz / 1000.0
            fspl = free_space_path_loss(cfg.frequency_mhz, cfg.distance_km)
            rain_db = rain_attenuation(frequency_ghz, cfg.distance_km, 25.0)
            assert fspl > 0 and rain_db >= 0

            # Step 4: Simplified link budget sanity check
            received_power_dbm = 30.0 + 2 * antenna_cfg.gain_dbi - (fspl + rain_db)
            assert -200 <= received_power_dbm <= 100
        finally:
            os.unlink(config_path)
            os.unlink(antenna_path)

    @pytest.mark.integration
    def test_run_simulation_fspl_end_to_end(self):
        """run_simulation returns a structured result for an FSPL scenario"""
        cfg = parse_simulation_config({
            "model": "fspl",
            "center_lat": 33.58831,
            "center_lng": -7.61138,
            "frequency": 900.0,
            "distance": 1.0,
            "area_radius_m": 100,
        })
        res = run_simulation(cfg)

        assert res["status"] == "ok"
        assert res["model"] == "fspl"
        assert res["path_loss_db"] > 0
        expected = 32.44 + 20 * __import__("numpy").log10(900.0) \
            + 20 * __import__("numpy").log10(1.0)
        assert abs(res["path_loss_db"] - expected) < 1e-6
        assert os.path.exists(res["map_file"])

    @pytest.mark.integration
    def test_pipeline_with_error_conditions(self):
        """Invalid configs raise during parsing instead of failing downstream"""
        invalid_config_data = {"frequency": -1000.0}
        config_path = _write_json(invalid_config_data)
        try:
            with pytest.raises((ValueError, AssertionError)):
                parse_simulation_config(config_path)
        finally:
            os.unlink(config_path)

    @pytest.mark.integration
    def test_pipeline_with_missing_components(self):
        """Minimal config parses; missing parameters fall back to defaults"""
        minimal_config_data = {"frequency": 3000.0, "distance": 5.0}
        config_path = _write_json(minimal_config_data)
        try:
            cfg = parse_simulation_config(config_path)
            assert cfg.frequency_mhz == 3000.0
            assert cfg.distance_km == 5.0
            assert cfg.tx_height_m == 30.0   # documented default
            assert cfg.rx_height_m == 1.5    # documented default
            assert cfg.model == "fspl"       # documented default
        finally:
            os.unlink(config_path)

    @pytest.mark.integration
    def test_pipeline_consistency_checks(self):
        """3 GHz over 1 km must match the closed-form FSPL (~101.98 dB)"""
        config_data = {"frequency": 3000.0, "distance": 1.0}
        config_path = _write_json(config_data)
        try:
            cfg = parse_simulation_config(config_path)
            actual_fspl = free_space_path_loss(cfg.frequency_mhz, cfg.distance_km)
            expected_fspl_approx = 102.0
            assert abs(actual_fspl - expected_fspl_approx) < 1.0
            assert actual_fspl > 0
        finally:
            os.unlink(config_path)
