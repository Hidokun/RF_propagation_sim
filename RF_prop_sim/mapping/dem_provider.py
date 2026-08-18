from __future__ import annotations

from typing import Optional

try:
    import rasterio
    from rasterio.errors import RasterioIOError
except ImportError:  # pragma: no cover
    rasterio = None
    RasterioIOError = Exception


class DemProvider:
    """Fetch point elevation from a local DEM file."""

    def __init__(self, dem_path: Optional[str] = None):
        self.dem_path = dem_path
        self._dataset = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def open(self):
        if self._dataset is not None:
            return self._dataset
        if self.dem_path is None:
            return None
        if rasterio is None:
            raise ImportError("rasterio is required for DemProvider. Install with 'pip install rasterio'.")

        try:
            self._dataset = rasterio.open(self.dem_path)
            return self._dataset
        except RasterioIOError as exc:
            raise FileNotFoundError(f"DEM file not found or unreadable: {self.dem_path}") from exc

    def get_elevation(self, lat: float, lon: float) -> float:
        """Return the elevation in meters for the given geographic coordinates."""
        dataset = self.open()
        if dataset is None:
            raise ValueError("DEM path is not configured for DemProvider.")

        row, col = dataset.index(lon, lat)
        band = dataset.read(1)
        value = band[row, col]
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(dataset.nodata) if dataset.nodata is not None else 0.0

    def close(self):
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None
