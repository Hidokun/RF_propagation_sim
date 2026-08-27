"""ITM reliability diagnostics (audit C1): kwx surfacing + fallback reasons
must reach compute_coverage_result["warnings"] for UI banners."""
import pytest

import propagation_model.itm_model as itm_model
from coverage_engine import compute_coverage_result
from propagation_model import free_space_path_loss


class TestCheckKwx:
    def test_below_threshold_returns_none(self):
        assert itm_model._check_kwx({"kwx": 0}, 5.0, 900.0, 30.0, 1.5) is None
        assert itm_model._check_kwx({"kwx": 2}, 5.0, 900.0, 30.0, 1.5) is None
        assert itm_model._check_kwx({}, 5.0, 900.0, 30.0, 1.5) is None

    @pytest.mark.parametrize("kwx", [3, 4])
    def test_flagged_levels_produce_message(self, kwx):
        msg = itm_model._check_kwx({"kwx": kwx}, 5.0, 900.0, 30.0, 1.5)
        assert msg is not None
        assert f"kwx={kwx}" in msg
        assert "out of model validity" in msg
        # Hook updated for the engine's snapshot diffing
        assert itm_model._LAST_KWX_WARNING == msg


class TestFallbackReasonHook:
    def test_missing_package_sets_reason(self, monkeypatch):
        """ImportError path must populate _LAST_FALLBACK_REASON."""
        def raise_import():
            raise ImportError("simulated missing itmlogic")
        monkeypatch.setattr(itm_model, "_load_itmlogic", raise_import)
        monkeypatch.setattr(itm_model, "_LAST_FALLBACK_REASON", None)

        result = itm_model.itm_path_loss(900.0, 5.0, 30.0, 1.5)

        # Fallback value is finite and >= FSPL baseline
        import math
        from propagation_model import free_space_path_loss
        assert isinstance(result, float) and not math.isnan(result)
        assert result >= free_space_path_loss(900.0, 5.0)
        assert itm_model._LAST_FALLBACK_REASON.startswith("ImportError")


class TestWarningsPlumbing:
    def test_fallback_warning_reaches_result(self, monkeypatch):
        """A failing ITM call inside the vector grid must surface in
        compute_coverage_result['warnings'] (audit C1 end-to-end)."""
        import numpy as np

        def boom(freq_mhz, distance_km, tx_height_m=30.0, rx_height_m=1.5,
                 terrain_type="average", surface_refractivity=None,
                 effective_earth_radius_factor=4/3, ground_permittivity=15.0,
                 ground_conductivity=0.005, terrain_profile_m=None):
            # Mimic the real wrapper's contract: on internal failure it sets
            # the diagnostic hook and RETURNS the approximate fallback.
            itm_model._LAST_FALLBACK_REASON = "forced:ValueError: boom"
            return free_space_path_loss(freq_mhz, distance_km) + 30.0

        monkeypatch.setattr("coverage_engine.itm_path_loss", boom)
        monkeypatch.setattr(itm_model, "_LAST_FALLBACK_REASON", None)

        res = compute_coverage_result(
            [{"name": "TX1", "lat": 33.58831, "lng": -7.61138,
              "frequency_mhz": 900.0, "tx_power_dbm": 40.0,
              "gain_dbi": 0.0, "height_m": 30.0, "nature": "transmitter"}],
            center_lat=33.58831, center_lng=-7.61138,
            box_size_m=2500.0, resolution_m=500.0,   # ring beyond 1 km exists
            model="itm", buildings_gdf=None)

        assert res is not None
        assert any("fell back" in w for w in res["warnings"]), res["warnings"]
        assert any("forced" in w for w in res["warnings"])

    def test_clean_run_has_empty_warnings(self):
        res = compute_coverage_result(
            [{"name": "TX1", "lat": 33.58831, "lng": -7.61138,
              "frequency_mhz": 900.0, "tx_power_dbm": 40.0,
              "gain_dbi": 0.0, "height_m": 30.0, "nature": "transmitter"}],
            center_lat=33.58831, center_lng=-7.61138,
            box_size_m=300.0, resolution_m=150.0,
            model="fspl", buildings_gdf=None)
        assert res is not None
        assert res["warnings"] == []
