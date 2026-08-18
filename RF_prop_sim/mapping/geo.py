from typing import Sequence, Tuple
import numpy as np

try:
    import pymap3d as pm
except ImportError:  # pragma: no cover
    pm = None


class GeoTransform:
    """Geospatial transformation utilities."""
    
    @staticmethod
    def geodetic_to_enu(lat: float, lon: float, h: float, origin: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Convert a single geodetic point to local ENU coordinates relative to an origin."""
        if pm is None:
            raise ImportError("pymap3d is required for geodetic_to_enu. Install with 'pip install pymap3d'.")
        lat0, lon0, h0 = origin
        e, n, u = pm.geodetic2enu(lat, lon, h, lat0, lon0, h0)
        return float(e), float(n), float(u)

    @staticmethod
    def geodetic_to_enu_array(
        lats: Sequence[float],
        lons: Sequence[float],
        hs: Sequence[float],
        origin: Tuple[float, float, float],
    ) -> np.ndarray:
        """Convert arrays of geodetic coordinates to ENU coordinates relative to an origin."""
        if pm is None:
            raise ImportError("pymap3d is required for geodetic_to_enu_array. Install with 'pip install pymap3d'.")

        lat0, lon0, h0 = origin
        lat_arr = np.asarray(lats, dtype=float)
        lon_arr = np.asarray(lons, dtype=float)
        h_arr = np.asarray(hs, dtype=float)

        e, n, u = pm.geodetic2enu(lat_arr, lon_arr, h_arr, lat0, lon0, h0)
        return np.vstack((np.asarray(e, dtype=float), np.asarray(n, dtype=float), np.asarray(u, dtype=float))).T


# Module-level functions for backward compatibility
def geodetic_to_enu(lat: float, lon: float, h: float, origin: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Convert a single geodetic point to local ENU coordinates relative to an origin."""
    return GeoTransform.geodetic_to_enu(lat, lon, h, origin)


def geodetic_to_enu_array(
    lats: Sequence[float],
    lons: Sequence[float],
    hs: Sequence[float],
    origin: Tuple[float, float, float],
) -> np.ndarray:
    """Convert arrays of geodetic coordinates to ENU coordinates relative to an origin."""
    return GeoTransform.geodetic_to_enu_array(lats, lons, hs, origin)
