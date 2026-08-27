"""
Longley-Rice Irregular Terrain Model (ITM) wrapper.

Uses the 'itmlogic' PyPI package (a port of Hufford's ITS model v1.2.2) when
available. Verified empirically against physical expectations: in this port,
lrprop's ``aref`` is the EXCESS attenuation over free space, so the final
median path loss is FSPL + max(aref, 0).

Call chain (point-to-point mode):
    qlrps()  -> setup constants (wn, gme, ens, zgnd)
    qlrpfl() -> horizon scan over the terrain profile pfl, then internal
                lrprop(0) initialization
    final    -> FSPL + aref

When itmlogic is missing, a documented approximate fallback is used.
"""

import logging
import numpy as np

from .empirical_models import free_space_path_loss

logger = logging.getLogger(__name__)

# Reasons for the most recent degraded result (test/diagnostic hooks that the
# coverage engine snapshots to surface reliability banners in the UI).
_LAST_FALLBACK_REASON = None
_LAST_KWX_WARNING = None


# Terrain profile sampling for synthetic (flat) paths: points per km
_PROFILE_POINTS_PER_KM = 20


def _load_itmlogic():
    """Import and return (qlrps, qlrpfl) or raise ImportError."""
    from itmlogic.preparatory_subroutines.qlrps import qlrps
    from itmlogic.preparatory_subroutines.qlrpfl import qlrpfl
    return qlrps, qlrpfl


def _check_kwx(prop, distance_km: float, frequency_mhz: float,
               tx_height_m: float, rx_height_m: float):
    """Audit M-2 helper: translate itmlogic's validity flag into a message.

    kwx severity (set by lrprop/qlrpfl):
      1 = freq/height/dist outside recommended range
      3 = grazing/horizon geometry anomaly
      4 = hard out-of-validity (overrides lower flags)
    Returns a human-readable warning string for kwx >= 3, else None.
    """
    kwx = int(prop.get("kwx", 0))
    if kwx < 3:
        return None
    msg = (f"itmlogic flagged low reliability (kwx={kwx}) for "
           f"{distance_km:.3f} km @ {frequency_mhz:.1f} MHz "
           f"(hg=[{tx_height_m:.1f}, {rx_height_m:.1f}] m) - result may be "
           f"out of model validity.")
    logging.getLogger(__name__).warning(msg)
    globals()["_LAST_KWX_WARNING"] = msg
    return msg


def _fallback_path_loss(
    frequency_mhz,
    distance_km,
    tx_height_m,
    rx_height_m,
    terrain_type,
):
    """Approximate terrain-aware estimator used when itmlogic is absent."""
    baseline = free_space_path_loss(frequency_mhz, distance_km)
    horizon_km = 3.57 * (np.sqrt(tx_height_m) + np.sqrt(rx_height_m))
    diffraction_penalty = max(0.0, 8.0 * np.log10(max(distance_km - horizon_km, 1.0)))
    terrain_penalty = {"average": 0.0, "hilly": 7.0, "mountainous": 14.0}.get(terrain_type, 3.0)
    return baseline + diffraction_penalty + terrain_penalty


def itm_path_loss(
    frequency_mhz,
    distance_km,
    tx_height_m,
    rx_height_m,
    terrain_type="average",
    surface_refractivity=None,
    effective_earth_radius_factor=4.0 / 3.0,
    ground_permittivity=15.0,
    ground_conductivity=0.005,
    terrain_profile_m=None,
):
    """Return ITM (Longley-Rice) median path loss in dB.

    Args:
        frequency_mhz: Carrier frequency in MHz (> 0).
        distance_km: Great-circle path length in km (> 0).
        tx_height_m: Transmitter antenna height above ground in meters.
        rx_height_m: Receiver antenna height above ground in meters.
        terrain_type: 'average' | 'hilly' | 'mountainous' (fallback tuning only;
            the real ITM derives terrain behavior from the elevation profile).
        surface_refractivity: Sea-level surface refractivity (N-units); default 301.
        effective_earth_radius_factor: Accepted for API compatibility; the real
            model derives curvature internally from refractivity.
        ground_permittivity: Relative ground permittivity (eps).
        ground_conductivity: Ground conductivity in S/m (sgm).
        terrain_profile_m: Optional list of elevations (m) sampled uniformly
            along the path, endpoints included. None => flat-earth profile.
    """
    # --- input validation (raise clean errors for bad types/values) ---
    try:
        frequency_mhz = float(frequency_mhz)
        distance_km = float(distance_km)
        tx_height_m = float(tx_height_m)
        rx_height_m = float(rx_height_m)
    except (TypeError, ValueError):
        raise TypeError("itm_path_loss arguments must be numeric")

    if frequency_mhz <= 0:
        raise ValueError("frequency_mhz must be positive")
    if distance_km <= 0:
        raise ValueError("distance_km must be positive")
    if tx_height_m < 0 or rx_height_m < 0:
        raise ValueError("heights must be non-negative")

    try:
        qlrps, qlrpfl = _load_itmlogic()
    except ImportError as exc:
        globals()["_LAST_FALLBACK_REASON"] = f"ImportError: {exc}"
        logger.warning("itmlogic package is not installed; using approximate "
                       "fallback (%s)", exc)
        return _fallback_path_loss(frequency_mhz, distance_km, tx_height_m,
                                   rx_height_m, terrain_type)

    try:
        # Setup constants; ipol=0 (horizontal), zsys=0 (sea-level reference)
        wn, gme, ens, zgnd = qlrps(
            frequency_mhz, 0.0,
            float(surface_refractivity) if surface_refractivity is not None else 301.0,
            0, float(ground_permittivity), float(ground_conductivity),
        )

        # Build terrain profile pfl = [n, delta_m, e0, ..., en] (elevations in m,
        # spacing in m). NOTE: pfl[0] must be an int for this port's indexing.
        distance_m = distance_km * 1000.0
        if terrain_profile_m is not None and len(terrain_profile_m) >= 2:
            elevations = [float(e) for e in terrain_profile_m]
            n = len(elevations) - 1
        else:
            n = int(min(max(distance_km * _PROFILE_POINTS_PER_KM, 20), 200))
            elevations = [0.0] * (n + 1)

        prop = {
            "wn": wn,
            "gme": gme,
            "ens": ens,
            "zgnd": zgnd,
            "hg": [tx_height_m, rx_height_m],
            "pfl": [n, distance_m / n] + elevations,
            "kwx": 0,
            "hgt": 0,
            "lbc": 0,
            "wscat": 0,
            "lvar": 1,
            "mdvarx": -1,   # keep default time/location variability setup
            "klimx": -1,    # keep default climate zone handling
        }

        prop = qlrpfl(prop)  # runs horizon scan + lrprop(0) init internally

        # Audit M-2: surface itmlogic's out-of-validity flags instead of
        # silently returning degraded numbers.
        _check_kwx(prop, distance_km, frequency_mhz, tx_height_m, rx_height_m)

        # aref is excess over free space in this port (verified empirically
        # against two-ray, knife-edge, and beyond-horizon regimes).
        excess_db = max(float(prop.get("aref", 0.0)), 0.0)
        return free_space_path_loss(frequency_mhz, distance_km) + excess_db

    except (KeyError, ValueError, IndexError, ZeroDivisionError,
            FloatingPointError) as exc:
        # Narrowed per audit M-1: expected upstream-port failures degrade to
        # the documented fallback WITH a logged reason; unexpected exception
        # types propagate so real bugs aren't masked.
        reason = f"{type(exc).__name__}: {exc}"
        globals()["_LAST_FALLBACK_REASON"] = reason
        logging.getLogger(__name__).warning(
            "itmlogic execution failed (%s); using approximate fallback.", reason)
        return _fallback_path_loss(frequency_mhz, distance_km, tx_height_m,
                                   rx_height_m, terrain_type)
