"""
Fetch real SRTM-based elevation data (AWS/OpenMapTiles Terrarium tiles) and
save as a GeoTIFF usable by RF_prop_sim's DemProvider.

Data source: https://registry.opendata.aws/terrain-tiles/ (open license,
no API key required). Elevation decoding for Terrarium PNGs:
    elev_m = R * 256 + G + B / 256 - 32768

Usage:
    python scripts/fetch_srtm_dem.py --west -7.6614 --south 33.5383 \
        --east -7.5614 --north 33.6383 --zoom 12 \
        --output data/dem/casablanca_srtm.tif

Note: output grid uses a linear lat/lon transform fitted over the requested
box; across regional boxes (<0.5 deg) mercator row compression introduces
negligible (<0.1%) spacing error.
"""

import argparse
import io
import math
import os
import sys
import urllib.request

import numpy as np


TILE_PX = 256


def lng_to_tile_x(lng, zoom):
    return int((lng + 180.0) / 360.0 * (2 ** zoom))


def lat_to_tile_y(lat, zoom):
    lat_rad = math.radians(lat)
    return int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * (2 ** zoom))


def lat_to_merc_row(lat_deg, zoom):
    """Global float pixel row (across all tiles) for a latitude."""
    lat_rad = math.radians(lat_deg)
    return (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * (2 ** zoom) * TILE_PX


def fetch_tile(zoom, x, y):
    url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{x}/{y}.png"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    from PIL import Image
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.asarray(img, dtype=np.float64)
    return arr[:, :, 0] * 256 + arr[:, :, 1] + arr[:, :, 2] / 256 - 32768


def main():
    ap = argparse.ArgumentParser(description="Download Terrarium DEM as GeoTIFF")
    ap.add_argument("--west", type=float, required=True)
    ap.add_argument("--south", type=float, required=True)
    ap.add_argument("--east", type=float, required=True)
    ap.add_argument("--north", type=float, required=True)
    ap.add_argument("--zoom", type=int, default=12,
                    help="Tile zoom; 12 ~= 31 m/px at Casablanca latitude")
    ap.add_argument("--output", default="casablanca_srtm.tif")
    args = ap.parse_args()

    x0 = lng_to_tile_x(args.west, args.zoom)
    x1 = lng_to_tile_x(args.east, args.zoom)
    y0 = lat_to_tile_y(args.north, args.zoom)   # north -> smaller tile y
    y1 = lat_to_tile_y(args.south, args.zoom)

    print(f"Fetching {(x1 - x0 + 1)}x{(y1 - y0 + 1)} tile(s), zoom {args.zoom} ...")

    parts_rows = []
    for ty in range(y0, y1 + 1):
        row_parts = [fetch_tile(args.zoom, tx, ty) for tx in range(x0, x1 + 1)]
        parts_rows.append(np.concatenate(row_parts, axis=1))
    mosaic = np.concatenate(parts_rows, axis=0)

    # Global pixel coordinates of the requested bounds. Pixel-center
    # convention (audit L-12): a pixel's coordinate is at its CENTER, so add
    # 0.5 px when mapping bounds to indices.
    deg_per_tile = 360.0 / (2 ** args.zoom)
    px_w = deg_per_tile / TILE_PX                      # degrees per px column
    col0_f = (args.west - (x0 * deg_per_tile - 180.0)) / px_w + 0.5
    col1_f = (args.east - (x0 * deg_per_tile - 180.0)) / px_w + 0.5
    row0_f = lat_to_merc_row(args.north, args.zoom) - y0 * TILE_PX + 0.5
    row1_f = lat_to_merc_row(args.south, args.zoom) - y0 * TILE_PX + 0.5

    c0, c1 = max(int(round(col0_f)), 0), min(int(round(col1_f)), mosaic.shape[1])
    r0, r1 = max(int(round(row0_f)), 0), min(int(round(row1_f)), mosaic.shape[0])
    crop = mosaic[r0:r1, c0:c1]

    if crop.size == 0:
        raise SystemExit("Crop produced an empty array; check bounds/zoom.")

    # Linear geotransform over the box (see module docstring note)
    height_px = crop.shape[0]
    px_h = (args.north - args.south) / height_px        # degrees per px row

    from rasterio.transform import from_origin
    import rasterio

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(
        args.output, "w", driver="GTiff",
        height=crop.shape[0], width=crop.shape[1],
        count=1, dtype="float32",
        crs="EPSG:4326", transform=from_origin(args.west, args.north, px_w, px_h),
    ) as dst:
        dst.write(crop.astype("float32"), 1)

    print(f"Wrote {args.output}: {crop.shape[1]}x{crop.shape[0]} px, "
          f"elev {crop.min():.0f}..{crop.max():.0f} m")


if __name__ == "__main__":
    sys.exit(main())
