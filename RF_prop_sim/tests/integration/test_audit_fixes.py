"""Integration pins for audit remediations (M-3/M-4/M-7/M-8/M-11)."""
import json
import os
import tempfile
import pytest

from RF_prop_sim.input_data_collection.ingestion import (
    parse_simulation_config,
    load_config_from_file,
)
from RF_prop_sim.mapping.dem_provider import DemProvider


# Project-root data/dem: integration -> tests -> RF_prop_sim -> repo root
_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", ".."))
CASABLANCA_TIF = os.path.join(_ROOT, "data", "dem", "casablanca_srtm.tif")


@pytest.mark.integration
class TestDemProviderGuards:
    def test_outside_bounds_raises_instead_of_wrapping(self):
        """Audit M-3: numpy negative-index wrap used to return WRONG elevations."""
        if not os.path.exists(CASABLANCA_TIF):
            pytest.skip("Casablanca DEM tile not present")
        with DemProvider(CASABLANCA_TIF) as dem:
            # Far outside the tile -> must raise, never return a wrapped value
            with pytest.raises(ValueError, match="outside DEM"):
                dem.get_elevation(60.0, 20.0)

    def test_inside_bounds_returns_finite(self):
        if not os.path.exists(CASABLANCA_TIF):
            pytest.skip("Casablanca DEM tile not present")
        with DemProvider(CASABLANCA_TIF) as dem:
            v = dem.get_elevation(33.58831, -7.61138)
        assert isinstance(v, float) and abs(v) < 1e4


@pytest.mark.integration
class TestReceiverCsvStrictness:
    def _csv(self, content):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_missing_columns_raise_loudly(self):
        # Two data rows -> multi-row receiver-grid path (single row = config).
        p = self._csv("Lat,LongX\n1.0,2.0\n3.0,4.0\n")   # no lng/longitude/long
        try:
            with pytest.raises(ValueError, match="lng"):
                parse_simulation_config(p)
        finally:
            os.unlink(p)

    def test_valid_rows_parse_with_case_insensitive_headers(self):
        p = self._csv("Lat,LonG,height_m\n33.5883,-7.6114,2.5\n33.5890,-7.6120,\n")
        try:
            cfg = parse_simulation_config(p)
            assert len(cfg.receivers) == 2
            assert cfg.receivers[0].height_m == pytest.approx(2.5)
            assert cfg.receivers[1].height_m == pytest.approx(1.5)   # default
        finally:
            os.unlink(p)

    def test_malformed_rows_dropped_with_tally(self, capsys):
        p = self._csv("lat,lng\n33.5883,-7.6114\nnot_a_number,-7.6\n")
        try:
            cfg = parse_simulation_config(p)
            assert len(cfg.receivers) == 1
            out = capsys.readouterr().out
            assert "dropped 1" in out
        finally:
            os.unlink(p)


@pytest.mark.integration
class TestReceiversWiredThroughRunSimulation:
    def test_cfg_receivers_produce_reports(self):
        """Audit M-8: parsed receivers must reach the output."""
        from RF_prop_sim.main import run_simulation
        from RF_prop_sim.input_data_collection.ingestion import ReceiverPoint

        cfg = parse_simulation_config({
            "model": "fspl",
            "center_lat": 33.58831,
            "center_lng": -7.61138,
            "frequency_mhz": 900.0,
            "distance_km": 1.0,
            "area_radius_m": 100,
        })
        cfg.receivers = [ReceiverPoint(lat=33.5890, lng=-7.6110, height_m=1.5)]
        res = run_simulation(cfg)
        reports = res.get("receiver_reports")
        assert reports and len(reports) == 1
        # Real value (never the -200 sentinel), and finite
        assert reports[0]["rssi_dbm"] > -200.0


@pytest.mark.integration
class TestReceiverWeatherForwarding:
    """Audit fix #1: run_simulation receiver reports must honor the user's
    weather/terrain configuration instead of engine kwarg defaults."""

    BASE = {
        "model": "fspl",
        "center_lat": 33.58831,
        "center_lng": -7.61138,
        "frequency_mhz": 900.0,
        "distance_km": 1.0,
        "area_radius_m": 100,
    }

    def _cfg(self, **over):
        from RF_prop_sim.input_data_collection.ingestion import (
            parse_simulation_config, ReceiverPoint)
        cfg = parse_simulation_config({**self.BASE, **over})
        cfg.receivers = [ReceiverPoint(lat=33.5890, lng=-7.6110, height_m=1.5)]
        return cfg

    @staticmethod
    def _rssi(res):
        reports = res.get("receiver_reports")
        assert reports, "receiver_reports missing"
        return reports[0]["rssi_dbm"]

    def _run(self, cfg, monkeypatch):
        """run_simulation with buildings stubbed away: guarantees the receiver
        is OUTDOOR (weather applies outdoors-only) and keeps tests offline."""
        monkeypatch.setattr("RF_prop_sim.main.download_buildings",
                            lambda *a, **k: None)
        from RF_prop_sim.main import run_simulation
        return run_simulation(cfg)

    def test_rain_zero_receiver_budget_equals_fspl(self, monkeypatch):
        dry = self._rssi(self._run(self._cfg(model="rain", rain_rate_mmh=0.0),
                                   monkeypatch))
        ref = self._rssi(self._run(self._cfg(model="fspl"), monkeypatch))
        # rain@0 dB/km is exactly its FSPL base -> budgets must coincide
        assert dry == pytest.approx(ref, abs=1e-6)

    def test_rain_heavy_attenuates_receivers(self, monkeypatch):
        # Rain attenuation scales as f^0.88: it is negligible at 900 MHz
        # (~0.003 dB/km), so probe at mmWave where the term is measurable.
        wet = self._rssi(self._run(self._cfg(model="rain", rain_rate_mmh=150.0,
                                             frequency_mhz=40000.0),
                                   monkeypatch))
        ref = self._rssi(self._run(self._cfg(model="fspl",
                                             frequency_mhz=40000.0),
                                   monkeypatch))
        assert wet < ref          # weather term strictly applied
        assert 0.0 < ref - wet < 5.0   # sane magnitude over a ~100 m link

    def test_combining_and_kwargs_reach_engine(self, monkeypatch):
        """combining + weather kwargs must be forwarded (not silently defaulted)."""
        captured = {}

        def fake_evaluate(rx, tx, model=None, combining="superposition",
                          buildings_gdf=None, **kw):
            captured.update(model=model, combining=combining, kw=dict(kw))
            return [{"name": rx[0]["name"], "lat": rx[0]["lat"],
                     "lng": rx[0]["lng"], "rssi_dbm": -70.0, "zone": "good",
                     "color": "#22c55e", "serving_antenna": tx[0]["name"],
                     "covered": True}]

        monkeypatch.setattr("coverage_engine.evaluate_receivers", fake_evaluate)
        cfg = self._cfg(model="rain", rain_rate_mmh=12.5,
                        relative_humidity=77.0, combining="best_server")
        self._run(cfg, monkeypatch)
        assert captured["model"] == "rain"
        assert captured["combining"] == "best_server"
        assert captured["kw"]["rain_rate_mmh"] == pytest.approx(12.5)
        assert captured["kw"]["relative_humidity"] == pytest.approx(77.0)
        assert captured["kw"]["tx_height_m"] == pytest.approx(cfg.tx_height_m)


@pytest.mark.integration
class TestCombiningAndFogConfigParsing:
    """Audit fixes #3+#4: combining field validated; legacy fog keys remapped."""

    def test_combining_accepted_and_validated(self, capsys):
        cfg = parse_simulation_config({"combining": "best_server"})
        assert cfg.combining == "best_server"
        cfg2 = parse_simulation_config({"combining": "bogus_mode"})
        out = capsys.readouterr().out
        assert cfg2.combining == "superposition"
        assert "unknown combining" in out.lower()

    def test_combining_default_superposition(self):
        assert parse_simulation_config(None).combining == "superposition"

    def test_legacy_fog_keys_map_to_real_field(self):
        cfg = parse_simulation_config({"fog_density": 0.2})
        assert cfg.fog_liquid_water_density_gm3 == pytest.approx(0.2)
        cfg2 = parse_simulation_config({"fog": 0.4})
        assert cfg2.fog_liquid_water_density_gm3 == pytest.approx(0.4)

    def test_dead_fog_fields_removed(self):
        cfg = parse_simulation_config(None)
        assert not hasattr(cfg, "fog_density")
        assert not hasattr(cfg, "fog_visibility_km")
