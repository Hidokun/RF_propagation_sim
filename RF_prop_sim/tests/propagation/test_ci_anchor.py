"""Closed-form anchor for the Close-In model (audit C-1).

Canonical CI: PL = 32.44 + 20lg(f_MHz) + 20lg(d0_km) + 10n*lg(d/d0).
With n=2 the model must collapse EXACTLY onto FSPL for ANY reference
distance d0 -- the property the historical +60 dB bug violated.
"""
import pytest

from coverage_engine import _base_loss_matrix
from propagation_model import close_in_path_loss, free_space_path_loss


@pytest.mark.propagation
@pytest.mark.parametrize("ref_distance_m", [1.0, 10.0, 100.0, 1000.0])
def test_ci_equals_fspl_at_n2_any_reference_distance(ref_distance_m):
    f, d = 900.0, 3.0
    ci = close_in_path_loss(f, d, reference_distance_m=ref_distance_m,
                            path_loss_exponent=2.0)
    fspl = free_space_path_loss(f, d)
    assert ci == pytest.approx(fspl, abs=1e-6), (
        f"CI(n=2, d0={ref_distance_m} m) must equal FSPL; got {ci:.3f} vs {fspl:.3f}"
    )


@pytest.mark.propagation
def test_ci_closed_form_nontrivial_exponent():
    f, d, d0, n = 900.0, 2.0, 100.0, 3.5
    expected = (32.44 + 20 * __import__("math").log10(f)
                + 20 * __import__("math").log10(d0 / 1000.0)
                + 10 * n * __import__("math").log10((d * 1000.0) / d0))
    assert close_in_path_loss(f, d, reference_distance_m=d0,
                              path_loss_exponent=n) == pytest.approx(expected, abs=1e-9)


@pytest.mark.propagation
def test_vector_twin_matches_canonical_ci():
    import numpy as np
    d = np.array([0.05, 1.0, 5.0])
    vec = _base_loss_matrix("ci", 900.0, d,
                            ref_distance_m=100.0, path_loss_exp=3.5)
    ref = np.array([close_in_path_loss(900.0, x, reference_distance_m=100.0,
                                       path_loss_exponent=3.5) for x in d])
    assert np.allclose(vec, ref, atol=1e-9)
