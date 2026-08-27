"""Shared builders for audit/regression tests across the remediation effort."""
import numpy as np


def synthetic_buildings(lng0, lat0, lng1, lat1, name="block"):
    """GeoDataFrame with one rectangular footprint."""
    import geopandas as gpd
    from shapely.geometry import box
    return gpd.GeoDataFrame(
        {"name": [name]},
        geometry=[box(lng0, lat0, lng1, lat1)],
        crs="EPSG:4326",
    )


def multi_block_buildings(origin_lng=-7.61160, origin_lat=33.58800,
                          n=6, dx=0.0008, dy=None):
    import geopandas as gpd
    from shapely.geometry import box
    dy = 0.0006 if dy is None else dy
    boxes = [box(origin_lng + i * dx, origin_lat,
                 origin_lng + i * dx + dx * 0.5, origin_lat + dy)
             for i in range(n)]
    return gpd.GeoDataFrame(geometry=boxes, crs="EPSG:4326")


class DeadTerrainSampler:
    """Stand-in mimicking a sampler whose DEM failed to load.

    `available` False; any sample_grid call is a test failure (the engines
    must not touch terrain when the sampler reports unavailable).
    """
    error = "synthetic dead sampler"

    @property
    def available(self) -> bool:
        return False

    def elevation(self, lat, lng):  # pragma: no cover - must never be called
        raise AssertionError("DeadTerrainSampler.elevation() called")

    def profile(self, *a, **k):  # pragma: no cover
        raise AssertionError("DeadTerrainSampler.profile() called")

    def sample_grid(self, *a, **k):  # pragma: no cover
        raise AssertionError("DeadTerrainSampler.sample_grid() called")


def tx(**overrides):
    """Canonical single-transmitter dict used across engine tests."""
    ant = {
        "name": "TX1", "lat": 33.58831, "lng": -7.61138,
        "frequency_mhz": 900.0, "tx_power_dbm": 40.0,
        "gain_dbi": 0.0, "height_m": 30.0, "nature": "transmitter",
    }
    ant.update(overrides)
    return ant
