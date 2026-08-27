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
        """Return the elevation in meters for the given geographic coordinates.

        Raises:
            ValueError: point lies outside the raster, or the sampled pixel is
                a nodata sentinel (previously these silently wrapped via
                numpy negative indexing / returned -32767 as real meters).
        """
        dataset = self.open()
        if dataset is None:
            raise ValueError("DEM path is not configured for DemProvider.")

        row, col = dataset.index(lon, lat)
        h, w = dataset.height, dataset.width
        # Guard BEFORE indexing: numpy would happily wrap negative or
        # overflowed indices to the opposite edge of the raster.
        if not (0 <= row < h and 0 <= col < w):
            raise ValueError(
                f"Point ({lat}, {lon}) is outside DEM coverage "
                f"(row={row}, col={col}, size={h}x{w})."
            )
        band = dataset.read(1)
        value = band[row, col]
        nodata = dataset.nodata
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"DEM pixel at ({lat}, {lon}) is not numeric.") from None
        if nodata is not None and value_f == float(nodata):
            raise ValueError(f"DEM pixel at ({lat}, {lon}) is nodata.")
        return value_f

    def close(self):
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None
