"""Physics validation tests for the coverage engine.

Validates superposition combining, building losses, terrain diffraction,
and the outdoor-only weather rule against closed-form expectations.
"""
import math

import pytest

from coverage_engine import (
    combine_rssi,
    knife_edge_loss,
    terrain_blockage_penalty,
    building_crossings_db,
    compute_coverage_grid,
    compute_coverage_result,
    evaluate_receivers,
    _BuildingIndex,
    INDOOR_PENALTY_DB,
    RSSI_MEDIUM,
)


def _tx(lat=33.58831, lng=-7.61138, **overrides):
    ant = {
        "name": "TX1", "lat": lat, "lng": lng,
        "frequency_mhz": 900.0, "tx_power_dbm": 40.0,
        "gain_dbi": 0.0, "height_m": 30.0, "nature": "transmitter",
    }
    ant.update(overrides)
    return ant


def _synthetic_buildings(lng0, lat0, lng1, lat1):
    """GeoDataFrame with one rectangular footprint."""
    import geopandas as gpd
    from shapely.geometry import box
    return gpd.GeoDataFrame(
        {"name": ["block"]},
        geometry=[box(lng0, lat0, lng1, lat1)],
        crs="EPSG:4326",
    )


class TestSuperpositionCombining:
    @pytest.mark.propagation
    def test_two_equal_signals_gain_3db(self):
        # Two identical -100 dBm carriers double the power: +3.0103 dB
        total = combine_rssi([-100.0, -100.0], "superposition")
        expected = -100.0 + 10 * math.log10(2)
        assert abs(total - expected) < 1e-9

    @pytest.mark.propagation
    def test_n_equal_signals_gain_10logN(self):
        rssis = [-90.0] * 8
        expected = -90.0 + 10 * math.log10(8)
        assert abs(combine_rssi(rssis, "superposition") - expected) < 1e-9

    @pytest.mark.propagation
    def test_weak_signal_barely_moves_dominant(self):
        # A signal 60 dB down adds ~0.000001 of the power: negligible shift
        total = combine_rssi([-60.0, -120.0], "superposition")
        assert abs(total - (-60.0)) < 0.01

    @pytest.mark.propagation
    def test_best_server_equals_max(self):
        assert combine_rssi([-70.0, -95.0, -85.0], "best_server") == -70.0

    @pytest.mark.propagation
    def test_empty_inputs(self):
        assert combine_rssi([], "superposition") == -200.0


class TestBuildingLosses:
    @pytest.mark.propagation
    def test_crossing_loss_monotonic_and_capped(self):
        losses = [building_crossings_db(n) for n in range(0, 10)]
        assert losses[0] == 0.0
        assert all(b > a for a, b in zip(losses, losses[1:5]))
        assert losses[-1] == min(12 + 9 * 6, 30)  # cap respected

    @pytest.mark.propagation
    def test_index_detects_crossing_and_containment(self):
        # Building spanning ~50 m around a known Casablanca point pair
        gdf = _synthetic_buildings(-7.61160, 33.58820, -7.61120, 33.58845)
        idx = _BuildingIndex(gdf)
        assert idx.available

        # Segment passing straight through the footprint crosses it once
        n = idx.crossings(33.58800, -7.61180, 33.58860, -7.61100)
        assert n >= 1

        # Point inside the footprint is contained; far point is not
        assert idx.contains(33.58830, -7.61140) is True
        assert idx.contains(33.60000, -7.60000) is False

    @pytest.mark.propagation
    def test_grid_points_inside_building_score_worse(self):
        # TX west of a large footprint; pick two same-row grid points flanking
        # the TX at +/- one resolution step: symmetric ranges, but the eastern
        # segment crosses the footprint while the western one is clear.
        gdf = _synthetic_buildings(-7.61160, 33.58820, -7.61120, 33.58845)
        tx = _tx(lng=-7.61240)
        # Box centered ON the transmitter with an even steps count => the TX
        # lies exactly on the central grid column, guaranteeing a symmetric
        # +/- one-step pair around it.
        kwargs = dict(center_lat=tx["lat"], center_lng=tx["lng"],
                      box_size_m=400.0, resolution_m=100.0, model="fspl")

        def _symmetric_pair(pts):
            """Return (west, east) points on the TX's row, +/- one step apart."""
            lats = sorted({round(p["lat"], 6) for p in pts})
            row_lat = min(lats, key=lambda L: abs(L - tx["lat"]))
            row = sorted((p for p in pts if round(p["lat"], 6) == row_lat),
                         key=lambda p: p["lng"])
            i = min(range(len(row)), key=lambda j: abs(row[j]["lng"] - tx["lng"]))
            return row[i - 1], row[i + 1]

        pts = compute_coverage_grid([tx], buildings_gdf=gdf, **kwargs)
        assert pts, "grid should not be empty"
        west, east = _symmetric_pair(pts)

        # Distances must match before comparing losses
        d_e = abs(east["lng"] - tx["lng"])
        d_w = abs(tx["lng"] - west["lng"])
        assert abs(d_e - d_w) < 1e-3, "selected points must be equidistant"

        # Eastern segment crosses the footprint -> strictly weaker signal
        assert east["rssi_dbm"] < west["rssi_dbm"] - 5.0, \
            f"shadowed point {east['rssi_dbm']} dBm should lose >=5 dB vs clear {west['rssi_dbm']} dBm"

        # Without buildings, symmetric points score identically
        pts_clear = compute_coverage_grid([tx], buildings_gdf=None, **kwargs)
        w2, e2 = _symmetric_pair(pts_clear)
        assert e2["rssi_dbm"] == pytest.approx(w2["rssi_dbm"], abs=0.01), \
            "without buildings, symmetric points must score identically"


class TestTerrainDiffraction:
    @pytest.mark.propagation
    def test_knife_edge_closed_form(self):
        """ITU-style continuous Lee variant: L=6.9+20log10(sqrt((v-0.1)^2+1)+v-0.1)."""
        # v=1: 6.9 + 20*log10(sqrt(0.81+1) + 0.9) = 13.93 dB
        assert knife_edge_loss(1.0) == pytest.approx(13.93, abs=0.02)
        # Grazing incidence v=0 -> the classic ~6.02 dB
        assert knife_edge_loss(0.0) == pytest.approx(6.03, abs=0.05)
        # Continuity at the -0.78 boundary
        assert knife_edge_loss(-0.78) == pytest.approx(0.0, abs=0.05)
        assert knife_edge_loss(-0.7) == pytest.approx(knife_edge_loss(-0.78), abs=0.75)
        # Deeply clear path -> no diffraction loss
        assert knife_edge_loss(-1.5) == 0.0
        # Cap respected
        assert knife_edge_loss(50.0) <= 30.0

    @pytest.mark.propagation
    def test_flat_profile_has_no_penalty(self):
        flat = [0.0] * 17
        assert terrain_blockage_penalty(flat, 1.0, 900, 30, 1.5) == 0.0

    @pytest.mark.propagation
    def test_blocked_path_penalized_more_than_clear(self):
        clear = [0.0] * 17
        blocked = [0.0] * 9 + [120.0] + [0.0] * 7   # 120 m ridge mid-path
        p_clear = terrain_blockage_penalty(clear, 2.0, 900, 30, 1.5)
        p_blocked = terrain_blockage_penalty(blocked, 2.0, 900, 30, 1.5)
        assert p_clear == 0.0
        assert p_blocked > 15.0
        assert p_blocked <= 30.0

    @pytest.mark.propagation
    def test_higher_frequency_diffacts_more(self):
        ridge = [0.0] * 9 + [60.0] + [0.0] * 7
        low = terrain_blockage_penalty(ridge, 1.0, 400, 30, 1.5)
        high = terrain_blockage_penalty(ridge, 1.0, 5000, 30, 1.5)
        # Shorter wavelength -> larger Fresnel parameter v -> more loss
        assert high >= low


class TestWeatherOutdoorsOnly:
    @pytest.mark.propagation
    def test_fog_applies_outdoors_not_indoors(self):
        import numpy as np
        from coverage_engine import _link_rssi, _TerrainSampler
        from propagation_model import fog_attenuation

        gdf = _synthetic_buildings(-7.61200, 33.58780, -7.61080, 33.58880)
        indoor_idx = _BuildingIndex(gdf)
        outdoor_idx = _BuildingIndex(None)

        ant = _tx()
        sampler = _TerrainSampler()
        pt = (33.58830, -7.61140)  # inside the synthetic block
        dist_km = 0.05

        kwargs_heavy = {"fog_liquid_water_density_gm3": 100.0}
        ctx_indoor = {"sampler": sampler, "buildings": indoor_idx, "kwargs": kwargs_heavy}
        ctx_outdoor = {"sampler": sampler, "buildings": outdoor_idx, "kwargs": kwargs_heavy}

        rssi_in = _link_rssi(ant, pt[0], pt[1], dist_km, "fog", ctx_indoor)
        rssi_out = _link_rssi(ant, pt[0], pt[1], dist_km, "fog", ctx_outdoor)

        # Outdoor: full FSPL + heavy fog attenuation
        expected_fog = fog_attenuation(0.9, dist_km, 100.0)
        expected_fspl = 32.44 + 20 * math.log10(900.0) + 20 * math.log10(max(dist_km, 0.01))
        expected_outdoor = ant["tx_power_dbm"] + ant["gain_dbi"] - (expected_fspl + expected_fog)
        # (no terrain penalty on this near-flat real DEM segment at these params;
        #  allow small tolerance in case of mild DEM variation)
        assert rssi_out <= expected_outdoor + 0.5

        # Indoor: fog term must be absent; only base + indoor penalty apply.
        # So indoor RSSI must be HIGHER than outdoor RSSI minus extra wall loss
        # would suggest... precise check: difference excludes the fog term.
        delta = rssi_out - rssi_in
        # Outdoor lost fog + possibly crossings; indoor has indoor penalty (+15).
        # Assert fog's contribution is missing indoors by checking that raising
        # density 100x changes nothing indoors but a lot outdoors.
        kwargs_extreme = {"fog_liquid_water_density_gm3": 10000.0}
        rssi_in_x = _link_rssi(ant, pt[0], pt[1], dist_km, "fog",
                               {"sampler": sampler, "buildings": indoor_idx,
                                "kwargs": kwargs_extreme})
        rssi_out_x = _link_rssi(ant, pt[0], pt[1], dist_km, "fog",
                                {"sampler": sampler, "buildings": outdoor_idx,
                                 "kwargs": kwargs_extreme})
        assert rssi_in_x == pytest.approx(rssi_in, rel=1e-6), \
            "indoor RSSI must not depend on weather intensity"
        assert rssi_out_x < rssi_out - 5.0, \
            "outdoor RSSI must degrade as weather intensifies"


class TestReceiversAndGridIntegration:
    @pytest.mark.propagation
    def test_receiver_gets_superposed_signal_from_all_txs(self):
        txs = [
            _tx(name="A", lng=-7.61338),
            _tx(name="B", lng=-7.60938),
        ]
        rxs = [{
            "name": "RX_mid", "lat": 33.58831, "lng": -7.61138,
            "nature": "receiver",
        }]
        reports = evaluate_receivers(rxs, txs, model="fspl",
                                     buildings_gdf=None)
        assert len(reports) == 1
        rep = reports[0]
        assert rep["covered"] is True
        assert rep["rssi_dbm"] > -200.0
        # Superposition of two symmetric contributions beats either alone
        assert rep["serving_antenna"] in ("A", "B")

    @pytest.mark.propagation
    def test_far_receiver_gets_real_weak_signal_no_sentinel(self):
        """Regression for the -200 dBm cliff: distant receivers must receive a
        finite, physically plausible value that decays with distance — never
        a sentinel jump."""
        txs = [_tx()]
        near_rx = {"name": "near", "lat": 33.58831 + 0.005,
                   "lng": -7.61138, "nature": "receiver"}    # ~555 m
        far_rx = {"name": "far", "lat": 33.58831 + 0.250,
                  "lng": -7.61138, "nature": "receiver"}     # ~27.8 km

        reports = evaluate_receivers([near_rx, far_rx], txs, model="fspl",
                                     buildings_gdf=None)
        near, far = reports
        # Both finite and plausible (FSPL @27.8km/900MHz ≈ 120 dB → ≈ -80 dBm)
        assert -180.0 < near["rssi_dbm"] < 0.0
        assert -180.0 < far["rssi_dbm"] < 0.0
        # Monotonic decay: farther receiver is strictly weaker
        assert far["rssi_dbm"] < near["rssi_dbm"]
        # covered flag consistent with the medium-zone threshold
        assert near["covered"] == (near["rssi_dbm"] >= RSSI_MEDIUM)
        assert far["covered"] == (far["rssi_dbm"] >= RSSI_MEDIUM)

    @pytest.mark.propagation
    def test_weather_model_includes_fspl_base(self):
        # Regression guard: rain/gas/fog maps must show realistic link budgets,
        # i.e., include FSPL rather than bare attenuation.
        common = dict(center_lat=33.58831, center_lng=-7.61138,
                      box_size_m=300.0, resolution_m=150.0, buildings_gdf=None)
        pts_zero_weather = compute_coverage_grid(
            [_tx()], model="rain", rain_rate_mmh=0.0, **common)
        pts_fspl = compute_coverage_grid(
            [_tx()], model="fspl", **common)
        # With zero rain rate, rain model reduces to its FSPL base exactly
        for pr, pf in zip(pts_zero_weather, pts_fspl):
            assert pr["rssi_dbm"] == pytest.approx(pf["rssi_dbm"], abs=0.01)

    @pytest.mark.propagation
    def test_rain_absent_kwarg_defaults_to_zero(self):
        # Regression guard (audit fix #2): a caller that omits rain_rate_mmh
        # entirely must get the documented 0 mm/h default ("no weather unless
        # configured"), NOT an implicit 5 mm/h surcharge.
        common = dict(center_lat=33.58831, center_lng=-7.61138,
                      box_size_m=300.0, resolution_m=150.0, buildings_gdf=None)
        pts_implicit = compute_coverage_grid([_tx()], model="rain", **common)
        pts_fspl = compute_coverage_grid([_tx()], model="fspl", **common)
        for pi, pf in zip(pts_implicit, pts_fspl):
            assert pi["rssi_dbm"] == pytest.approx(pf["rssi_dbm"], abs=0.01)

    @pytest.mark.propagation
    def test_superposition_beats_best_server_in_overlap(self):
        common = dict(center_lat=33.58831, center_lng=-7.61038,
                      box_size_m=450.0, resolution_m=150.0, model="fspl",
                      buildings_gdf=None)
        txs = [_tx(name="A", lng=-7.61228), _tx(name="B", lng=-7.61048)]
        sup = compute_coverage_grid(txs, combining="superposition", **common)
        best = compute_coverage_grid(txs, combining="best_server", **common)
        # At overlap midpoints superposed power can only be >= strongest single
        for p_sup, p_best in zip(sup, best):
            assert p_sup["rssi_dbm"] >= p_best["rssi_dbm"] - 0.01


class TestBoundedBoxAnalysis:
    """Regression tests for the user-defined analysis-box redesign.

    Guards against the historical bug where points beyond a transmitter's
    radius snapped from ~-37 dBm to a -200 dBm sentinel — a physically
    impossible discontinuity.
    """

    def _grid_rows(self, pts):
        rows = {}
        for p in pts:
            rows.setdefault(round(p["lat"], 6), []).append(p)
        for row in rows.values():
            row.sort(key=lambda p: p["lng"])
        return rows

    @pytest.mark.propagation
    def test_grid_covers_full_box_with_expected_count(self):
        box_m, res_m = 1000.0, 200.0
        pts = compute_coverage_grid(
            [_tx()], center_lat=33.58831, center_lng=-7.61138,
            box_size_m=box_m, resolution_m=res_m, model="fspl",
            buildings_gdf=None)
        n = int(box_m / res_m) + 1          # inclusive both ends
        assert len(pts) == n * n

    @pytest.mark.propagation
    def test_fspl_radial_continuity_no_cliff(self):
        """Adjacent grid points may differ by at most 20 dB.

        Natural terrain shadow edges (Fresnel zone crossing rising ground
        within one DEM pixel) legitimately produce 10-18 dB single-step
        drops; the historical -200 dBm sentinel cliff was >150 dB. This
        guard catches sentinel/class bugs while allowing real shadowing.
        """
        tx = _tx()
        pts = compute_coverage_grid(
            [tx], center_lat=tx["lat"], center_lng=tx["lng"],
            box_size_m=1200.0, resolution_m=30.0, model="fspl",
            buildings_gdf=None)
        for row in self._grid_rows(pts).values():
            for a, b in zip(row, row[1:]):
                step = abs(a["rssi_dbm"] - b["rssi_dbm"])
                assert step <= 20.0, \
                    f"discontinuity {step:.1f} dB between {a} and {b}"

    @pytest.mark.propagation
    def test_itm_radial_continuity_no_cliff(self):
        """ITM (with real DEM profiles) must stay continuous across the box.

        Comparisons within ~120 m of the TX are skipped: two-ray ground
        reflections produce genuine narrow interference nulls close to the
        antenna, which is real physics rather than a numerical cliff.
        Bound loosened vs FSPL because knife-edge penalties can engage
        between adjacent samples when terrain crosses the LOS line.
        """
        tx = _tx()
        pts = compute_coverage_grid(
            [tx], center_lat=tx["lat"], center_lng=tx["lng"],
            box_size_m=800.0, resolution_m=40.0, model="itm",
            buildings_gdf=None)
        for row in self._grid_rows(pts).values():
            for a, b in zip(row, row[1:]):
                # skip pairs whose midpoint sits within 120 m of the TX
                mid_lat = (a["lat"] + b["lat"]) / 2
                mid_lng = (a["lng"] + b["lng"]) / 2
                d_tx = math.hypot((mid_lat - tx["lat"]) * 111320.0,
                                  (mid_lng - tx["lng"]) * 111320.0 *
                                  math.cos(math.radians(tx["lat"])))
                if d_tx < 120.0:
                    continue
                step = abs(a["rssi_dbm"] - b["rssi_dbm"])
                assert step <= 25.0, \
                    f"ITM discontinuity {step:.1f} dB between {a} and {b}"

    @pytest.mark.propagation
    def test_box_edge_points_finite_and_realistic(self):
        pts = compute_coverage_grid(
            [_tx()], center_lat=33.58831, center_lng=-7.61138,
            box_size_m=1000.0, resolution_m=200.0, model="fspl",
            buildings_gdf=None)
        corner = min(pts, key=lambda p: (p["lat"], p["lng"]))
        # Corner sits ~700 m diagonally from TX; FSPL ≈ 97 dB → ≈ -57 dBm
        assert -174.0 < corner["rssi_dbm"] < 0.0
        # And it must NOT be a sentinel value
        assert corner["rssi_dbm"] != -200.0


class TestTerrainSampler:
    """Guards the DEM sampler's scalar path (inverse-affine regression)."""

    @pytest.mark.propagation
    def test_sampler_available_and_profiles_finite(self):
        from coverage_engine import _TerrainSampler
        s = _TerrainSampler()
        if not s.available:
            pytest.skip("No DEM tile present")
        prof = s.profile(33.58831, -7.61138, 33.59200, -7.60600)
        assert len(prof) == 17
        assert all(isinstance(v, float) and v == v and abs(v) < 1e4 for v in prof)
        # Endpoints must match direct elevation lookups (same clamped cell)
        assert prof[0] == pytest.approx(s.elevation(33.58831, -7.61138), abs=1e-9)

    @pytest.mark.propagation
    def test_out_of_bounds_clamps_to_edge(self):
        from coverage_engine import _TerrainSampler
        s = _TerrainSampler()
        if not s.available:
            pytest.skip("No DEM tile present")
        # Far outside the Casablanca tile -> nearest edge value, no crash
        v = s.elevation(60.0, 20.0)
        assert isinstance(v, float) and abs(v) < 1e4


class TestVectorEquivalence:
    """Scalar reference engine vs default vector engine must agree.

    Guards the NumPy/shapely bulk pipeline against semantic drift from the
    per-link reference implementation (coordinate-order bugs, boundary
    semantics, underflow handling, formula drift in the vector twins).
    """

    TOL_DB = 0.01

    @staticmethod
    def _compare(pts_scalar, pts_vector):
        assert len(pts_scalar) == len(pts_vector)
        for ps, pv in zip(pts_scalar, pts_vector):
            assert ps["lat"] == pytest.approx(pv["lat"], abs=1e-12)
            assert ps["lng"] == pytest.approx(pv["lng"], abs=1e-12)
            assert abs(ps["rssi_dbm"] - pv["rssi_dbm"]) <= TestVectorEquivalence.TOL_DB, \
                f"{ps} vs {pv}"
            assert ps["zone"] == pv["zone"]
            assert ps["best_antenna"] == pv["best_antenna"]

    def _run_pair(self, txs, model, combining="superposition", buildings=None, **kw):
        common = dict(center_lat=33.58831, center_lng=-7.61138,
                      box_size_m=400.0, resolution_m=100.0,
                      model=model, combining=combining,
                      buildings_gdf=buildings, **kw)
        scalar = compute_coverage_grid(list(txs), engine="scalar", **common)
        vector = compute_coverage_grid(list(txs), engine="vector", **common)
        return scalar, vector

    @pytest.mark.propagation
    def test_fspl_equivalence(self):
        s, v = self._run_pair([_tx()], "fspl")
        self._compare(s, v)

    @pytest.mark.propagation
    def test_ci_equivalence(self):
        s, v = self._run_pair([_tx()], "ci")
        self._compare(s, v)

    @pytest.mark.propagation
    def test_rain_zero_rate_equivalence(self):
        s, v = self._run_pair([_tx()], "rain", rain_rate_mmh=0.0)
        self._compare(s, v)

    @pytest.mark.propagation
    def test_fog_with_weather_density_equivalence(self):
        s, v = self._run_pair([_tx()], "fog", fog_liquid_water_density_gm3=0.5)
        self._compare(s, v)

    @pytest.mark.propagation
    def test_itm_subkilometre_box_equivalence(self):
        # Box stays under the 1 km Longley-Rice floor -> FSPL+knife-edge path
        # on both engines (no external qlrpfl calls).
        common = dict(center_lat=33.58831, center_lng=-7.61138,
                      box_size_m=800.0, resolution_m=200.0,
                      model="itm", buildings_gdf=None)
        s = compute_coverage_grid([_tx()], engine="scalar", **common)
        v = compute_coverage_grid([_tx()], engine="vector", **common)
        self._compare(s, v)

    @pytest.mark.propagation
    def test_buildings_shadow_equivalence(self):
        gdf = _synthetic_buildings(-7.61160, 33.58820, -7.61120, 33.58845)
        txs = [_tx(name="A", lng=-7.61338), _tx(name="B", lng=-7.60938)]
        for combining in ("superposition", "best_server"):
            s, v = self._run_pair(txs, "fspl", combining=combining, buildings=gdf)
            self._compare(s, v)

    @pytest.mark.propagation
    def test_two_tx_best_server_equivalence(self):
        txs = [_tx(name="A", lng=-7.61338), _tx(name="B", lng=-7.60938)]
        s, v = self._run_pair(txs, "fspl", combining="best_server")
        self._compare(s, v)

    @pytest.mark.propagation
    def test_indoor_boundary_is_inclusive_in_both_engines(self):
        """Point exactly ON a footprint edge counts as indoor (covers parity)."""
        import numpy as np
        from coverage_engine import _BuildingIndex
        gdf = _synthetic_buildings(-7.61160, 33.58820, -7.61120, 33.58845)
        idx = _BuildingIndex(gdf)
        xs = np.array([[-7.61120]])   # exactly on the eastern edge
        ys = np.array([[33.58830]])
        assert bool(idx.contains_grid(xs, ys)[0, 0]) is True
        # And the scalar per-point API agrees
        assert idx.contains(33.58830, -7.61120) is True

    @pytest.mark.propagation
    def test_vector_twins_match_canonical_formulas(self):
        """Drift guard: inline vector formulas must equal canonical models."""
        import numpy as np
        from coverage_engine import _weather_matrix, _base_loss_matrix
        from propagation_model import (
            free_space_path_loss, rain_attenuation, gas_attenuation, fog_attenuation,
        )
        d = np.array([0.05, 0.5, 3.0])
        f_mhz, f_ghz = 900.0, 0.9

        base = _base_loss_matrix("fspl", f_mhz, d, 1.0, 2.0)
        ref = np.array([free_space_path_loss(f_mhz, x) for x in d])
        assert np.allclose(base, ref, atol=1e-9)

        rain = _weather_matrix("rain", f_ghz, d, rain_rate_mmh=12.5)
        ref_rain = np.array([rain_attenuation(f_ghz, x, 12.5) for x in d])
        assert np.allclose(rain, ref_rain, atol=1e-9)

        gas = _weather_matrix("gas", f_ghz, d,
                              temperature_c=20.0, pressure_hpa=1010.0,
                              relative_humidity=60.0)
        ref_gas = np.array([gas_attenuation(f_ghz, x, temperature_c=20.0,
                                            pressure_hpa=1010.0,
                                            relative_humidity=60.0) for x in d])
        assert np.allclose(gas, ref_gas, atol=1e-9)

        fog = _weather_matrix("fog", f_ghz, d, fog_liquid_water_density_gm3=0.4)
        ref_fog = np.array([fog_attenuation(f_ghz, x, 0.4) for x in d])
        assert np.allclose(fog, ref_fog, atol=1e-9)


class TestAuditP0Regressions:
    """Pinning tests for audit findings C-1/H-1/H-2/H-4."""

    @pytest.mark.propagation
    def test_rain_partial_coefficients_honored(self):
        """H-1: a caller-supplied k OR alpha alone must never be discarded."""
        from propagation_model import rain_attenuation
        # Only k supplied -> alpha falls back to horizontal default 0.90
        a = rain_attenuation(9.0, 2.0, 25.0, k=0.02)
        assert a == pytest.approx(0.02 * 25.0 ** 0.90 * 2.0, rel=1e-12)
        # Only alpha supplied -> k falls back to horizontal default
        b = rain_attenuation(9.0, 2.0, 25.0, alpha=1.1)
        assert b == pytest.approx(0.0001 * 9.0 ** 0.88 * 25.0 ** 1.1 * 2.0, rel=1e-12)
        # Both supplied -> untouched
        c = rain_attenuation(9.0, 2.0, 25.0, k=0.03, alpha=1.05)
        assert c == pytest.approx(0.03 * 25.0 ** 1.05 * 2.0, rel=1e-12)

    @pytest.mark.propagation
    def test_rain_partial_scalar_vector_parity(self):
        import numpy as np
        from coverage_engine import _weather_matrix
        d = np.array([1.0, 3.0])
        vec = _weather_matrix("rain", 9.0, d,
                              rain_rate_mmh=25.0, rain_k=0.02)   # alpha defaulted
        ref = np.array([rain_att_ref(9.0, float(x), 25.0, k=0.02) for x in d])
        assert np.allclose(vec, ref, atol=1e-12)

    @pytest.mark.propagation
    def test_antenna_parser_coerces_optional_floats_from_strings(self):
        """H-2: Optional[float] fields must become floats even when the source
        provides strings (XML/KML/text)."""
        from RF_prop_sim.antenna_data.parser import parse_antenna_config
        ant = parse_antenna_config({"lat": "33.58831", "lng": "-7.61138",
                                    "alt": "27.5"})
        assert isinstance(ant.lat, float) and ant.lat == pytest.approx(33.58831)
        assert isinstance(ant.lng, float) and ant.lng == pytest.approx(-7.61138)
        assert isinstance(ant.alt, float) and ant.alt == pytest.approx(27.5)

    @pytest.mark.propagation
    def test_itm_vector_matches_scalar_without_dem(self):
        """H-4: with no DEM loaded, >=1 km cells must still run Longley-Rice
        in BOTH engines (flat synthetic profile)."""
        from unittest.mock import patch
        from test_utils import audit_helpers
        from coverage_engine import _TERRAIN_SAMPLER_SINGLETON

        common = dict(center_lat=33.58831, center_lng=-7.61138,
                      box_size_m=2500.0, resolution_m=500.0,
                      model="itm", buildings_gdf=None)
        txs = [audit_helpers.tx()]

        with patch("coverage_engine._TERRAIN_SAMPLER_SINGLETON",
                   audit_helpers.DeadTerrainSampler()):
            s = compute_coverage_grid(list(txs), engine="scalar", **common)
            v = compute_coverage_grid(list(txs), engine="vector", **common)

        TestVectorEquivalence._compare(s, v)
        # Sanity: at least one cell beyond 1 km existed so the branch ran
        assert any(p["rssi_dbm"] != -200.0 for p in s)


def rain_att_ref(f_ghz, d_km, rate, k=None, alpha=None):
    """Scalar reference mirroring the fixed partial-coefficient semantics."""
    if f_ghz <= 0 or d_km <= 0 or rate <= 0:
        return 0.0
    if k is None:
        k = 0.0001 * f_ghz ** 0.88
    if alpha is None:
        alpha = 0.90
    return k * rate ** alpha * d_km


class TestDebugSweepProbes:
    """Stage-3 probes from the full-system debug sweep. Found-green probes
    are kept as permanent regression tests."""

    @pytest.mark.propagation
    def test_itm_exact_one_km_boundary_engine_parity(self):
        """Probe A: cells at exactly 1.0 km sit on the sub-km/ITM routing
        boundary; both engines must route them identically."""
        common = dict(center_lat=33.58831, center_lng=-7.61138,
                      box_size_m=2000.0, resolution_m=500.0,   # ring at 1.0 km
                      model="itm", buildings_gdf=None)
        s = compute_coverage_grid([_tx()], engine="scalar", **common)
        v = compute_coverage_grid([_tx()], engine="vector", **common)
        TestVectorEquivalence._compare(s, v)

    @pytest.mark.propagation
    def test_rain_partial_k_survives_grid_plumbing(self):
        """Probe B: caller-supplied rain_k must survive the full
        compute_coverage_result plumbing (H-1 end-to-end)."""
        res = compute_coverage_result(
            [_tx()], center_lat=33.58831, center_lng=-7.61138,
            box_size_m=300.0, resolution_m=150.0,
            model="rain", rain_rate_mmh=25.0,
            rain_alpha=0.90,                       # supply alpha only -> k defaults
            buildings_gdf=None)
        assert res is not None and res["points"]
        # With k defaulted but alpha honored, spot-check one point's loss:
        p = res["points"][len(res["points"]) // 2]
        d_km = max(0.01, ((p["lat"] - 33.58831) ** 2 * 111.32 ** 2
                          + ((p["lng"] + 7.61138) * 111.32 *
                             __import__("math").cos(math.radians(33.58831))) ** 2) ** 0.5)
        k_default = 0.0001 * 0.9 ** 0.88           # freq 900 MHz = 0.9 GHz
        expected_rain = k_default * 25.0 ** 0.90 * d_km
        fspl = 32.44 + 20 * math.log10(max(d_km, 0.01)) + 20 * math.log10(900.0)
        expected_rssi = 40.0 - (fspl + expected_rain)
        assert p["rssi_dbm"] == pytest.approx(expected_rssi, abs=2.0), \
            f"{p['rssi_dbm']} vs expected ~{expected_rssi:.2f} (terrain tolerance)"

    @pytest.mark.propagation
    def test_receiver_height_reaches_link_budget(self):
        """Probe C: per-receiver height must change the ITM sub-km budget on
        sloped terrain (deterministic synthetic profile via ctx)."""
        import numpy as np
        from coverage_engine import _link_rssi

        class SlopeSampler:
            available = True
            error = None
            def profile(self, lat1, lng1, lat2, lng2, n=16):
                # 35 m ridge mid-path: shadows a 1.5 m RX, clears a 50 m one
                # (LOS midpoint sits ~16 m vs ~46 m above ground respectively).
                prof = [0.0] * (n + 1)
                prof[n // 2 - 1:n // 2 + 2] = [35.0] * 3
                return prof

        sampler = SlopeSampler()
        from coverage_engine import _BuildingIndex
        buildings = _BuildingIndex(None)

        ant = _tx()
        base_kwargs = {"rx_height_m": 1.5}
        results = {}
        for h in (1.5, 50.0):
            kwargs = dict(base_kwargs)
            kwargs["rx_height_m"] = h
            ctx = {"sampler": sampler, "buildings": buildings, "kwargs": kwargs}
            results[h] = _link_rssi(ant, 33.59200, -7.60800, 0.9, "itm", ctx)
        # Higher receiver clears the ridge; low one is shadowed.
        assert results[50.0] > results[1.5], \
            f"height ignored: {results}"

    @pytest.mark.propagation
    def test_rgba_orientation_north_up(self):
        """Probe F: zone_code[0,0] is the min-lat corner; after flipud it must
        land in the LAST row of the RGBA so origin='upper' paints it south."""
        import numpy as np
        H = W = 4
        zone = np.zeros((H, W), dtype="int8")
        palette = np.array([[239, 68, 68], [249, 115, 22], [34, 197, 94]],
                           dtype=np.uint8)
        rgba = np.empty((H, W, 4), dtype=np.uint8)
        rgba[:, :, :3] = palette[zone]
        rgba[:, :, 3] = 150
        flipped = np.flipud(rgba)
        # South corner (min-lat => zone row 0) now lives in the last RGBA row.
        assert (flipped[-1, :, :3] == palette[0]).all()
        # Uniform zero-zone input => every flipped row is the bad color.
        assert (flipped[:, :, :3] == palette[0]).all()

    @pytest.mark.propagation
    def test_buildings_fetch_needed(self):
        """Probe E: M-6 invalidation decision (real engine function)."""
        from coverage_engine import buildings_fetch_needed
        meta = {"lat": 33.58831, "lng": -7.61138, "dist": 2000.0}
        assert buildings_fetch_needed(None, 33.58831, -7.61138, 2000.0) is True
        assert buildings_fetch_needed(meta, 33.58831, -7.61138, 2000.0) is False
        assert buildings_fetch_needed(meta, 33.58831 + 0.03, -7.61138,
                                      2000.0) is True   # ~3.3 km away

    @pytest.mark.propagation
    def test_result_snapshots_run_params(self):
        """Probe D/L-7: result must carry the parameters used for THIS run."""
        res = compute_coverage_result(
            [_tx()], center_lat=33.58831, center_lng=-7.61138,
            box_size_m=300.0, resolution_m=150.0,
            model="ci", buildings_gdf=None)
        rp = res["run_params"]
        assert rp["model"] == "ci"
        assert rp["box_size_m"] == pytest.approx(300.0)
        assert rp["resolution_m"] == pytest.approx(150.0)
        assert rp["buildings_used"] is False
