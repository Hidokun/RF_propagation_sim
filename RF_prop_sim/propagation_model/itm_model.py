"""
Longley-Rice Irregular Terrain Model (ITM) wrapper.
Relies on the 'itmlogic' python package when available.
"""

import numpy as np
from .empirical_models import free_space_path_loss


def _build_itm_property_dict(
    frequency_mhz,
    distance_km,
    tx_height_m,
    rx_height_m,
    terrain_type,
    surface_refractivity,
    effective_earth_radius_factor,
    ground_permittivity,
    ground_conductivity,
):
    return {
        "d": float(distance_km),
        "f": float(frequency_mhz),
        "frequency": float(frequency_mhz),
        "hg": float(tx_height_m),
        "he": float(rx_height_m),
        "gme": float(effective_earth_radius_factor),
        "ens": float(surface_refractivity) if surface_refractivity is not None else 301.0,
        "em": float(ground_permittivity),
        "sm": float(ground_conductivity),
        "pm": 1013.25,
        "temp": 15.0,
        "mdp": 0,
        "dmin": 0.001,  # Minimum distance (km) - ITM requirement
        "dlsa": 0.0,    # Delta left side angle - ITM requirement
        "wscat": 0,     # Wet surface category - ITM requirement
        "dl": [0.0, 0.0],  # Delta left array - ITM requirement (typically [left_angle, right_angle])
        "terrain_type": terrain_type,
    }


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
):
    """Return ITM path loss, using itmlogic when available."""
    try:
        import importlib
        lrprop_module = importlib.import_module("itmlogic.lrprop")
    except ImportError:
        print("Warning: itmlogic package is not installed. Falling back to an approximate terrain-aware propagation estimator.")
        baseline = free_space_path_loss(frequency_mhz, distance_km)
        horizon_km = 3.57 * (np.sqrt(tx_height_m) + np.sqrt(rx_height_m))
        diffraction_penalty = max(0.0, 8.0 * np.log10(max(distance_km - horizon_km, 1.0)))
        terrain_penalty = {"average": 0.0, "hilly": 7.0, "mountainous": 14.0}.get(terrain_type, 3.0)
        return baseline + diffraction_penalty + terrain_penalty

    prop = _build_itm_property_dict(
        frequency_mhz,
        distance_km,
        tx_height_m,
        rx_height_m,
        terrain_type,
        surface_refractivity,
        effective_earth_radius_factor,
        ground_permittivity,
        ground_conductivity,
    )

    try:
        if hasattr(lrprop_module, "qlrpfl"):
            result = lrprop_module.qlrpfl(prop)
            # Handle numpy arrays and scalars, as well as regular lists/tuples
            if hasattr(result, '__len__') and not isinstance(result, (str, bytes)):
                try:
                    # It's array-like, take the first element
                    result = result[0] if len(result) > 0 else result
                except TypeError:
                    pass
            # If result is already a scalar, use it directly
            return float(result)
        if hasattr(lrprop_module, "lrprop"):
            result = lrprop_module.lrprop(distance_km, prop)
            # lrprop returns the modified property dictionary
            if isinstance(result, dict):
                return float(result.get("aref", free_space_path_loss(frequency_mhz, distance_km)))
            elif isinstance(result, (int, float)):
                return float(result)
            else:
                return float(prop.get("aref", free_space_path_loss(frequency_mhz, distance_km)))
    except Exception as exc:
        print(f"Warning: itmlogic execution failed: {exc}")
        # Fallback to approximate method
        baseline = free_space_path_loss(frequency_mhz, distance_km)
        horizon_km = 3.57 * (np.sqrt(tx_height_m) + np.sqrt(rx_height_m))
        diffraction_penalty = max(0.0, 8.0 * np.log10(max(distance_km - horizon_km, 1.0)))
        terrain_penalty = {"average": 0.0, "hilly": 7.0, "mountainous": 14.0}.get(terrain_type, 3.0)
        return baseline + diffraction_penalty + terrain_penalty