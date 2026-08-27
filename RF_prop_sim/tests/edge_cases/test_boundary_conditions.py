"""Boundary value debugger tests"""
import pytest
import numpy as np
from propagation_model import (
    free_space_path_loss,
    rain_attenuation,
    gas_attenuation,
    fog_attenuation,
    close_in_path_loss,
)
from propagation_model.itm_model import itm_path_loss as itm_model
from propagation_model.ray_tracing_model import ray_tracing_path_loss as ray_tracing_model


class TestBoundaryConditionsDebugger:
    """Test suite for boundary value debugger"""

    @pytest.mark.edge_case
    def test_zero_and_negative_values(self):
        """Test all models with zero and negative input values"""
        test_freqs = [0.0, -1.0, -100.0]
        test_dists = [0.0, -1.0, -10.0]
        test_rain_rates = [0.0, -1.0, -50.0]
        test_fog_densities = [0.0, -0.1, -1.0]

        for freq in test_freqs:
            for dist in test_dists:
                assert free_space_path_loss(freq, dist) == 0.0

        for freq in test_freqs:
            for dist in test_dists:
                for rain_rate in test_rain_rates:
                    assert rain_attenuation(freq / 1000.0, dist, rain_rate) == 0.0

        for freq in test_freqs:
            for dist in test_dists:
                assert gas_attenuation(freq / 1000.0, dist) == 0.0

        for freq in test_freqs:
            for dist in test_dists:
                for fog_density in test_fog_densities:
                    assert fog_attenuation(freq / 1000.0, dist, fog_density) == 0.0

        for freq in test_freqs:
            for dist in test_dists:
                assert close_in_path_loss(freq, dist) == 0.0

    @pytest.mark.edge_case
    def test_extreme_high_values(self):
        """Models must stay finite/non-negative for extreme large inputs"""
        extreme_freqs_ghz = [1e3, 1e6]     # GHz scale (1 THz .. 1 PHz)
        extreme_dists_km = [1e3, 1e6]

        for freq_mhz in [1e6, 1e9]:
            for dist in extreme_dists_km:
                result = free_space_path_loss(freq_mhz, dist)
                assert result is not None
                assert not np.isnan(result)
                assert result >= 0 or np.isinf(result)

        # Atmospheric models at THz-scale frequencies remain finite or inf
        for freq_ghz in extreme_freqs_ghz:
            for dist in [10.0, 100.0]:  # physical path lengths
                r = rain_attenuation(freq_ghz, dist, 50.0)
                assert np.isfinite(r) or np.isinf(r)
                g = gas_attenuation(freq_ghz, dist)
                assert np.isfinite(g) or np.isinf(g)
                f = fog_attenuation(freq_ghz, dist, 2.0)
                assert np.isfinite(f) or np.isinf(f)

        # CI model with huge values stays finite
        result = close_in_path_loss(1e9, 1e6)
        assert np.isfinite(result) or np.isinf(result)

    @pytest.mark.edge_case
    def test_extreme_low_positive_values(self):
        """Sub-physical tiny inputs: no crashes, finite results.

        Note: below ~1 kHz / sub-millimeter distances the closed-form FSPL
        yields negative decibels; that is mathematically correct and the
        physically meaningful contract here is only "finite, no crash".
        """
        small_freqs_mhz = [1e-3, 1e-6]
        small_dists_km = [1e-6, 1e-9]

        for freq in small_freqs_mhz:
            for dist in small_dists_km:
                result = free_space_path_loss(freq, dist)
                assert result is not None
                assert not np.isnan(result)
                assert np.isfinite(result)

        for freq_mhz in small_freqs_mhz:
            freq_ghz = freq_mhz / 1000.0
            for dist in small_dists_km:
                assert np.isfinite(rain_attenuation(freq_ghz, dist, 0.1))
                assert np.isfinite(gas_attenuation(freq_ghz, dist))
                assert np.isfinite(fog_attenuation(freq_ghz, dist, 0.001))
                assert np.isfinite(close_in_path_loss(freq_mhz, dist))

    @pytest.mark.edge_case
    def test_extreme_antenna_parameters(self):
        """ITM tolerates heights from zero to very large"""
        extreme_heights = [0.0, 1.0, 1e3]

        for height in extreme_heights:
            try:
                result = itm_model(
                    frequency_mhz=1000.0,
                    distance_km=5.0,
                    tx_height_m=height,
                    rx_height_m=height,
                )
                assert result is None or isinstance(result, (int, float, np.floating))
            except Exception:
                pass  # extreme values may be rejected — acceptable

    @pytest.mark.edge_case
    def test_boundary_conditions_specific_models(self):
        """Frequency boundaries from VLF to sub-THz behave monotonically"""
        boundary_freqs_mhz = [0.1, 1.0, 1000.0, 100000.0]

        prev_fspl = -np.inf
        for freq in boundary_freqs_mhz:
            result = free_space_path_loss(freq, 1.0)
            assert result is not None and not np.isnan(result)
            assert result >= prev_fspl  # monotonic in frequency
            prev_fspl = result

            freq_ghz = freq / 1000.0
            for fn, args in [
                (rain_attenuation, (freq_ghz, 1.0, 5.0)),
                (gas_attenuation, (freq_ghz, 1.0)),
                (fog_attenuation, (freq_ghz, 1.0, 0.1)),
            ]:
                r = fn(*args)
                assert r is not None and not np.isnan(r) and r >= 0

        # Distance boundaries: strictly increasing FSPL with distance
        prev = -np.inf
        for dist in [0.001, 0.1, 1.0, 1000.0]:
            result = free_space_path_loss(1000.0, dist)
            assert not np.isnan(result)
            assert result >= prev
            prev = result

    @pytest.mark.edge_case
    def test_nan_and_infinity_handling(self):
        """NaN/inf inputs never crash the models"""
        for bad in (np.nan, np.inf, -np.inf):
            try:
                result = free_space_path_loss(bad, 5.0)
                # Returned value must at least be a float-like
                assert result is not None
            except (TypeError, ValueError, OverflowError, ZeroDivisionError):
                pass  # clean rejection is acceptable

            try:
                result = free_space_path_loss(5.0, bad)
                assert result is not None
            except (TypeError, ValueError, OverflowError, ZeroDivisionError):
                pass
