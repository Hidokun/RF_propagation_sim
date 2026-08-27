"""Temporary benchmark: scalar vs vector engine + render payload sizes."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RF_prop_sim'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from coverage_engine import compute_coverage_grid, zone_statistics

def synth_buildings():
    import geopandas as gpd
    from shapely.geometry import box
    boxes = [box(-7.61160+i*0.0008, 33.58800, -7.61120+i*0.0008, 33.58860) for i in range(6)]
    return gpd.GeoDataFrame(geometry=boxes, crs="EPSG:4326")

TX = {"name": "TX1", "lat": 33.58831, "lng": -7.61138, "frequency_mhz": 900.0,
      "tx_power_dbm": 40.0, "gain_dbi": 12.0, "height_m": 30.0, "nature": "transmitter"}
GDF = synth_buildings()

def bench(label, fn, n=3):
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        pts = fn()
        best = min(best, time.perf_counter() - t0)
    ms_per_link = best / len(pts) * 1000 if pts else float("nan")
    print(f"{label:<58} {best*1000:9.1f} ms   ({len(pts):>6} pts, {ms_per_link:.3f} ms/link)")
    return best

print("=" * 100)
for box, res in [(2000, 25), (5000, 50)]:
    n_side = int(round(box / res)) + 1
    print(f"--- box {box} m @ {res} m  -> {n_side}x{n_side} = {n_side*n_side} links ---")
    for model in ("fspl", "ci"):
        for bld, bname in ((None, "no-bld"), (GDF, "w/-bld")):
            common = dict(antennas=[TX], center_lat=33.58831, center_lng=-7.61138,
                          box_size_m=float(box), resolution_m=float(res),
                          model=model, buildings_gdf=bld)
            bench(f"{model:>4} scalar {bname}", lambda: compute_coverage_grid(engine="scalar", **common))
            bench(f"{model:>4} vector {bname}", lambda: compute_coverage_grid(engine="vector", **common))

# Render payload comparison (old circles vs raster+lattice)
pts = compute_coverage_grid(antennas=[TX], center_lat=33.58831, center_lng=-7.61138,
                            box_size_m=2000.0, resolution_m=25.0, model="fspl",
                            engine="vector", buildings_gdf=None)
H = W = int(round(2000 / 25)) + 1
stride = max(1, int(np.ceil((max(H, W) - 1) / 23.0)))
lattice = len(range(0, H, stride)) * len(range(0, W, stride))
print("=" * 100)
print(f"render objects @2km/25m: OLD {len(pts)} CircleMarkers -> NEW 1 ImageOverlay + {lattice} hover dots")
