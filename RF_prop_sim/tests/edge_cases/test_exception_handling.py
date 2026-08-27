"""Exception handling debugger tests"""
import pytest
import json
import os
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

from propagation_model import (
    free_space_path_loss,
    rain_attenuation,
    gas_attenuation,
    fog_attenuation,
    close_in_path_loss,
)
from propagation_model.itm_model import itm_path_loss as itm_model
from propagation_model.ray_tracing_model import ray_tracing_path_loss as ray_tracing_model
from RF_prop_sim.input_data_collection.ingestion import parse_simulation_config
from RF_prop_sim.antenna_data.parser import parse_antenna_config
from RF_prop_sim.mapping.location_service import Geocoder


def _write_json(data):
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(data, f)
    f.close()
    return f.name


class TestExceptionHandlingDebugger:
    """Test suite for exception handling debugger"""

    @pytest.mark.edge_case
    def test_invalid_types_handled_or_raised_cleanly(self):
        """String inputs either raise clean errors or degrade to 0 — never corrupt state"""
        # FSPL with strings raises TypeError from numeric comparison/coercion
        # OR returns 0.0 if implementation coerces; both are acceptable contracts,
        # but the error type must be a standard exception.
        try:
            result = free_space_path_loss("invalid", 5.0)
            assert result == 0.0  # degraded gracefully
        except TypeError:
            pass  # raised cleanly

        try:
            result = rain_attenuation("invalid", 5.0, 10.0)
            assert result == 0.0
        except TypeError:
            pass

    @pytest.mark.edge_case
    def test_exception_propagation_through_pipeline(self):
        """Invalid JSON surfaces as an exception from the config parser"""
        bad_json_path = _write_json_raw("{ invalid json content")
        try:
            with pytest.raises((ValueError, json.JSONDecodeError)):
                parse_simulation_config(bad_json_path)
        finally:
            if os.path.exists(bad_json_path):
                os.unlink(bad_json_path)

    @pytest.mark.edge_case
    def test_models_return_zero_for_out_of_domain_inputs(self):
        """Out-of-domain inputs degrade to 0.0 rather than raising"""
        assert free_space_path_loss(-1000.0, 5.0) == 0.0
        assert free_space_path_loss(3000.0, -5.0) == 0.0
        assert rain_attenuation(-1.0, 5.0, 25.0) == 0.0
        assert rain_attenuation(1.0, -5.0, 25.0) == 0.0
        assert rain_attenuation(1.0, 5.0, -1.0) == 0.0
        # Valid calls still work after out-of-domain ones (no corrupted state)
        assert free_space_path_loss(1000.0, 5.0) > 0
        assert rain_attenuation(1.0, 5.0, 25.0) > 0

    @pytest.mark.edge_case
    def test_itm_model_exception_handling(self):
        """ITM handles invalid inputs predictably (fallback or clean error)"""
        with pytest.raises((ValueError, TypeError)):
            itm_model(frequency_mhz="invalid", distance_km=5.0,
                      tx_height_m=10.0, rx_height_m=10.0)

    @pytest.mark.edge_case
    def test_ray_tracing_exception_handling(self):
        """Ray tracing handles invalid inputs predictably"""
        with pytest.raises((ValueError, TypeError, RuntimeError)):
            ray_tracing_model("invalid", [0, 0, 10], [100, 0, 1.5])

    @pytest.mark.edge_case
    def test_location_service_degrades_on_failure(self):
        """Geocoding failures degrade to None / fallback, never raw crashes"""
        service = Geocoder(api_key="fake-key-for-testing")
        fake_client = MagicMock()
        fake_client.geocode.side_effect = ConnectionError("Network unavailable")

        with patch("RF_prop_sim.mapping.location_service.googlemaps.Client",
                   return_value=fake_client):
            result = service.geocode("Somewhere")
        assert result is None

        # Without any key, documented fallback kicks in
        keyless = Geocoder(api_key=None)
        assert keyless.geocode("") is not None

    @pytest.mark.edge_case
    def test_resource_cleanup_on_validation_error(self):
        """A failed parse leaves global state intact for subsequent parses"""
        good_path = _write_json({"frequency": 3000.0, "distance": 5.0})
        bad_path = _write_json({"frequency": -1000.0, "distance": 5.0})
        try:
            with pytest.raises(ValueError):
                parse_simulation_config(bad_path)

            # Prior valid config still parses identically after the failure
            cfg = parse_simulation_config(good_path)
            assert cfg.frequency_mhz == 3000.0
        finally:
            for p in (good_path, bad_path):
                if os.path.exists(p):
                    os.unlink(p)


def _write_json_raw(content):
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    f.write(content)
    f.close()
    return f.name
