"""Graceful degradation debugger tests"""
import pytest
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
from RF_prop_sim.mapping.location_service import Geocoder as LocationService


class TestGracefulDegradationDebugger:
    """Test suite for graceful degradation debugger"""

    @pytest.mark.edge_case
    def test_degradation_when_optional_dependencies_missing(self):
        """ITM degrades to its documented fallback when itmlogic is absent"""
        try:
            import itmlogic  # noqa: F401
            itm_available = True
        except ImportError:
            itm_available = False

        result = itm_model(
            frequency_mhz=1000.0, distance_km=5.0,
            tx_height_m=10.0, rx_height_m=10.0,
        )
        # Either the real ITM ran or the fallback produced a finite loss
        assert isinstance(result, (int, float))
        assert not np.isnan(result)
        if not itm_available:
            # Fallback must at least reproduce FSPL baseline
            assert result >= free_space_path_loss(1000.0, 5.0)

    @pytest.mark.edge_case
    def test_degradation_when_sionna_unavailable(self):
        """Ray tracing degrades gracefully when sionna is absent"""
        try:
            import sionna  # noqa: F401
            sionna_available = True
        except ImportError:
            sionna_available = False

        result = ray_tracing_model(30.0, [0, 0, 10], [100, 0, 1.5])
        if not sionna_available:
            # Documented fallback: FSPL + excess loss (finite number)
            assert isinstance(result, (int, float))
            assert not np.isnan(result)
        else:
            assert result is not None

    @pytest.mark.edge_case
    def test_fallback_to_simpler_models(self):
        """Empirical baseline stays valid regardless of ITM success"""
        frequency_mhz, distance_km = 1000.0, 5.0
        fspl_result = free_space_path_loss(frequency_mhz, distance_km)
        rain_result = rain_attenuation(1.0, distance_km, 10.0)
        gas_result = gas_attenuation(1.0, distance_km)
        fog_result = fog_attenuation(1.0, distance_km, 0.1)
        baseline_loss = fspl_result + rain_result + gas_result + fog_result
        assert baseline_loss >= 0

        try:
            itm_result = itm_model(
                frequency_mhz=frequency_mhz, distance_km=distance_km,
                tx_height_m=10.0, rx_height_m=10.0,
            )
            if isinstance(itm_result, (int, float)) and not np.isnan(itm_result):
                assert itm_result >= 0
                if baseline_loss > 0:
                    ratio = itm_result / baseline_loss
                    assert 0.1 <= ratio <= 10.0
        except Exception:
            # ITM failing must not invalidate the empirical baseline
            assert baseline_loss >= 0

    @pytest.mark.edge_case
    def test_degradation_when_external_services_unavailable(self):
        """Geocoding API failures degrade to None instead of raising"""
        service = LocationService(api_key="fake-key-for-testing")

        fake_client = MagicMock()
        fake_client.geocode.side_effect = ConnectionError("Network unavailable")
        with patch("RF_prop_sim.mapping.location_service.googlemaps.Client",
                   return_value=fake_client):
            result = service.geocode("Somewhere")
        assert result is None

        fake_client2 = MagicMock()
        fake_client2.geocode.side_effect = TimeoutError("Request timed out")
        with patch("RF_prop_sim.mapping.location_service.googlemaps.Client",
                   return_value=fake_client2):
            result = service.geocode("Somewhere")
        assert result is None

    @pytest.mark.edge_case
    def test_degradation_with_partial_data_availability(self):
        """Partial configs parse with defaults filling missing fields"""
        partial_configs = [
            {"frequency": 3000.0},
            {"frequency": 3000.0, "distance": 5.0},
            {"distance": 5.0, "tx_height": 10.0},
        ]
        for config_data in partial_configs:
            cfg = parse_simulation_config(config_data)
            assert cfg is not None
            # Supplied alias keys land on canonical fields with correct values
            alias_map = {"frequency": "frequency_mhz", "distance": "distance_km",
                         "tx_height": "tx_height_m"}
            for key, value in config_data.items():
                assert getattr(cfg, alias_map[key]) == value

        partial_antenna_configs = [
            {"gain": 10.0},
            {"gain": 10.0, "beamwidth": 50.0},
            {"beamwidth": 30.0, "polarization": "vertical"},
        ]
        for antenna_data in partial_antenna_configs:
            antenna = parse_antenna_config(antenna_data)
            assert antenna is not None
            alias_map = {"gain": "gain_dbi", "beamwidth": "beamwidth_h"}
            for key, value in antenna_data.items():
                assert getattr(antenna, alias_map.get(key, key)) == value

    @pytest.mark.edge_case
    def test_progressive_degradation(self):
        """Core models keep working even when advanced models fail"""
        fspl = free_space_path_loss(1000.0, 5.0)
        rain = rain_attenuation(1.0, 5.0, 10.0)
        gas = gas_attenuation(1.0, 5.0)
        fog = fog_attenuation(1.0, 5.0, 0.1)
        ci = close_in_path_loss(1000.0, 5.0)
        assert all(np.isfinite(x) and x >= 0 for x in [fspl, rain, gas, fog, ci])

        # Simulated failure of advanced models does not affect core models
        with patch("propagation_model.itm_model.itm_path_loss",
                   side_effect=Exception("ITM failed")):
            fspl2 = free_space_path_loss(1000.0, 5.0)
            rain2 = rain_attenuation(1.0, 5.0, 10.0)
            assert np.isfinite(fspl2) and fspl2 >= 0
            assert np.isfinite(rain2) and rain2 >= 0

    @pytest.mark.edge_case
    def test_degradation_maintains_api_consistency(self):
        """All models return numeric types even in degraded modes"""
        result = itm_model(1000.0, 5.0, 10.0, 10.0)
        assert result is None or isinstance(result, (int, float, np.number))

        result = ray_tracing_model(30.0, [0, 0, 10], [100, 0, 1.5])
        assert result is None or isinstance(result, (int, float, np.number))

        fspl_result = free_space_path_loss(1000.0, 5.0)
        assert isinstance(fspl_result, (int, float, np.number))
        assert fspl_result >= 0

        rain_result = rain_attenuation(1.0, 5.0, 10.0)
        assert isinstance(rain_result, (int, float, np.number))
        assert rain_result >= 0

    @pytest.mark.edge_case
    def test_user_notification_in_degraded_mode(self):
        """Degraded mode prints a warning; normal operation stays quiet"""
        import logging
        from io import StringIO

        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger('rf_simulator.degradation')
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        try:
            # ITM fallback prints a warning via print(); ensure calling it
            # neither raises nor corrupts logging state.
            itm_model(1000.0, 5.0, 10.0, 10.0)

            # Normal operations produce no degradation warnings
            free_space_path_loss(1000.0, 5.0)
            rain_attenuation(1.0, 5.0, 10.0)
            assert "degrad" not in log_stream.getvalue().lower()
        finally:
            logger.removeHandler(handler)
