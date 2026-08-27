"""
coverage_engine.py

Multi-antenna RF coverage grid computation with physics-aware corrections.

For every grid point the engine computes, per covering transmitter:
    RSSI = tx_power_dBm + gain_dBi - path_loss_dB

where path_loss stacks, in order:
    1. Base propagation loss   (fspl / ci / itm / sionna-style excess)
    2. Weather attenuation     (rain / gas / fog) - OUTDOOR points only;
                                  weather models automatically gain an FSPL
                                  base so their maps show real link budgets
    3. Terrain diffraction     (DEM-sampled knife-edge loss w/ earth bulge,
                                or real terrain profiles fed to ITM)
    4. Building losses         (+12 dB first wall crossed, +6 dB each next,
                                capped; +15 dB indoor penetration)

Multi-transmitter combining (selectable):
    - "superposition" (default): powers sum in the linear domain,
      RSSI_total = 10*log10( sum(10^(RSSI_i/10)) )  -- how real receivers
      measure combined signals.
    - "best_server": strongest transmitter wins (classic handover maps).

Zone thresholds:
    "good"   — RSSI >= -80 dBm   (green)
    "medium" — RSSI >= -95 dBm   (orange)
    "bad"     — RSSI <  -95 dBm   (red)

Only antennas with nature "transmitter" radiate; receivers are evaluated via
evaluate_receivers().
"""

import json
import math
from typing import List, Dict, Any, Optional

from propagation_model import (
    free_space_path_loss,
    rain_attenuation,
    gas_attenuation,
    fog_attenuation,
    close_in_path_loss,
    itm_path_loss,
)

# RSSI thresholds for zone classification
RSSI_GOOD = -80   # dBm
RSSI_MEDIUM = -95  # dBm

# Zone to color mapping
ZONE_COLORS = {
    "good": "#22c55e",   # green
    "medium": "#f97316", # orange
    "bad": "#ef4444",    # red
}

WEATHER_MODELS = {"rain", "gas", "fog"}
GEOMETRY_MODELS = {"fspl", "ci", "itm", "sionna"}

# Building loss constants (urban clutter approximation)
FIRST_CROSSING_DB = 12.0   # first wall penetrated
EXTRA_CROSSING_DB = 6.0    # each additional wall
MAX_CROSSING_DB = 30.0     # deep-street saturation
INDOOR_PENALTY_DB = 15.0   # receiver inside a building

# Terrain constants
EARTH_RADIUS_KM = 6371.0
PROFILE_SAMPLES = 16       # DEM samples per link


# ──────────────────────────────────────────────────────────────────────────────
# Terrain sampling
# ──────────────────────────────────────────────────────────────────────────────

class _TerrainSampler:
    """Reads the DEM raster ONCE into memory; serves fast elevation lookups."""

    def __init__(self):
        self._arr = None
        self._bounds = None
        # Linear inverse-transform coefficients (col = a*x + b*y + c,
        # row = d*x + e*y + f) cached for vectorized sampling.
        self._inv = None
        self.error = None                    # reason terrain is unavailable, if any
        try:
            from mapping.terrain_builder import resolve_dem_path
            dem_path = resolve_dem_path()
            if dem_path:
                import logging
                import warnings
                import rasterio
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with rasterio.open(dem_path) as ds:
                        arr = ds.read(1).astype("float64")
                        inv = ~ds.transform
                        self._inv = (inv.a, inv.b, inv.c, inv.d, inv.e, inv.f)
                        self._bounds = ds.bounds
                        nodata = ds.nodata
                # Nodata holes: fill with the scene MEDIAN, not 0.0 — a zero
                # fill fabricates sea-level canyons that manufacture spurious
                # knife-edge losses (audit M-4). Log how many cells were fixed.
                import numpy as np
                if nodata is not None:
                    holes = int(np.count_nonzero(arr == float(nodata)))
                    if holes:
                        median = float(np.median(arr[arr != float(nodata)])) \
                            if np.count_nonzero(arr != float(nodata)) else 0.0
                        arr[arr == float(nodata)] = median
                        logging.getLogger(__name__).info(
                            "DEM '%s': filled %d nodata cells with scene median %.1f m",
                            dem_path, holes, median)
                self._arr = arr
        except Exception as exc:  # noqa: BLE001 - degrade, but LOUDLY and with reason
            import logging
            self.error = f"{type(exc).__name__}: {exc}"
            self._arr = None
            logging.getLogger(__name__).warning(
                "Terrain sampling disabled (%s); coverage will run flat-earth.", self.error)

    @property
    def available(self) -> bool:
        return self._arr is not None

    def elevation(self, lat: float, lng: float) -> float:
        """Elevation in meters; clamps to nearest edge outside coverage; 0 if no DEM."""
        if self._arr is None:
            return 0.0
        _, _, right, top = self._bounds.left, self._bounds.bottom, self._bounds.right, self._bounds.top
        lng_c = min(max(lng, self._bounds.left + 1e-9), right - 1e-9)
        lat_c = min(max(lat, self._bounds.bottom + 1e-9), top - 1e-9)
        # Inverse affine coefficients (col = a*x + b*y + c, row = d*x + e*y + f)
        a, b, c, d, e, f = self._inv
        col = a * lng_c + b * lat_c + c
        row = d * lng_c + e * lat_c + f
        r = int(min(max(row, 0), self._arr.shape[0] - 1))
        c_idx = int(min(max(col, 0), self._arr.shape[1] - 1))
        return float(self._arr[r, c_idx])

    def profile(self, lat1, lng1, lat2, lng2, n: int = PROFILE_SAMPLES) -> List[float]:
        """Ground elevations sampled uniformly along the great-ish path."""
        return [
            self.elevation(lat1 + (lat2 - lat1) * i / n,
                           lng1 + (lng2 - lng1) * i / n)
            for i in range(n + 1)
        ]

    def sample_grid(self, lat1, lng1, lat2_mat, lng_mat, n_ts=None):
        """Vectorized profiles: elevations along TX->point links for EVERY point.

        Returns array of shape (H, W, n+1) matching the scalar `profile()`
        semantics exactly (same clamp/truncation rules).
        """
        import numpy as np
        if self._arr is None:
            return None
        if n_ts is None:
            n_ts = PROFILE_SAMPLES
        H, W = lat2_mat.shape
        a, b, c, d, e, f = self._inv
        left, bottom, right, top = (self._bounds.left, self._bounds.bottom,
                                    self._bounds.right, self._bounds.top)
        out = np.empty((H, W, n_ts + 1), dtype="float64")
        for k in range(n_ts + 1):
            t = k / n_ts
            lat_k = lat1 + (lat2_mat - lat1) * t
            lng_k = lng1 + (lng_mat - lng1) * t
            lng_c = np.clip(lng_k, left + 1e-9, right - 1e-9)
            lat_c = np.clip(lat_k, bottom + 1e-9, top - 1e-9)
            col = a * lng_c + b * lat_c + c
            row = d * lng_c + e * lat_c + f
            r = np.clip(row.astype("int64"), 0, self._arr.shape[0] - 1)
            cc = np.clip(col.astype("int64"), 0, self._arr.shape[1] - 1)
            out[:, :, k] = self._arr[r, cc]
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Building indexing
# ──────────────────────────────────────────────────────────────────────────────

class _BuildingIndex:
    """Spatial index over OSM building footprints (degree-space, valid <~2 km)."""

    def __init__(self, buildings_gdf=None):
        self._geoms = []
        self._tree = None
        if buildings_gdf is None or len(buildings_gdf) == 0:
            return
        try:
            from shapely.geometry import LineString, Point as ShapelyPoint
            from shapely.strtree import STRtree
            geoms = []
            for geom in buildings_gdf.geometry:
                if geom is None or geom.is_empty:
                    continue
                # MultiPolygons -> individual parts for accurate counting
                if geom.geom_type == "Polygon":
                    geoms.append(geom)
                elif geom.geom_type == "MultiPolygon":
                    geoms.extend(list(geom.geoms))
            if not geoms:
                return
            self._geoms = geoms
            self._tree = STRtree(geoms)
            self._LineString = LineString
            self._Point = ShapelyPoint
        except Exception:
            self._geoms = []
            self._tree = None

    @property
    def available(self) -> bool:
        return self._tree is not None and len(self._geoms) > 0

    @property
    def polygons(self) -> list:
        """Underlying polygon list (for bulk vectorized predicates)."""
        return self._geoms

    def crossings(self, lat1, lng1, lat2, lng2) -> int:
        """Number of distinct building polygons intersected by the TX->point segment."""
        if not self.available:
            return 0
        line = self._LineString([(lng1, lat1), (lng2, lat2)])
        candidates = self._tree.query(line)
        hit = 0
        for idx in candidates:
            g = self._geoms[idx]
            try:
                if g.intersects(line):
                    hit += 1
            except Exception:
                continue
        return hit

    def contains(self, lat, lng) -> bool:
        """True if the point falls inside any building footprint."""
        if not self.available:
            return False
        pt = self._Point(lng, lat)
        candidates = self._tree.query(pt)
        for idx in candidates:
            try:
                if self._geoms[idx].covers(pt):
                    return True
            except Exception:
                continue
        return False

    # ── Bulk vectorized variants (shapely 2.x C-backed) ───────────────────────

    def crossings_grid(self, tx_lat: float, tx_lng: float, xs, ys):
        """Wall-crossing counts for segments TX->every (xs, ys) point.

        Equivalent to calling crossings() per point, executed as ONE bulk
        STRtree predicate over an array of LineStrings.
        `xs`/`ys` are the x(=lng)/y(=lat) coordinate matrices.
        Returns int array shaped like xs.
        """
        import numpy as np
        if not self.available:
            return np.zeros(xs.shape, dtype="int64")
        import shapely
        n = xs.size
        x1 = np.full(n, tx_lng, dtype="float64")
        y1 = np.full(n, tx_lat, dtype="float64")
        coords = np.stack([
            np.column_stack([x1, y1]),
            np.column_stack([xs.ravel(), ys.ravel()]),
        ], axis=1)                                   # (N, 2, 2)
        lines = shapely.linestrings(coords)
        res = self._tree.query(lines, predicate="intersects")
        if len(res[0]) == 0:
            return np.zeros(xs.shape, dtype="int64")
        counts = np.bincount(res[0], minlength=n)
        return counts.reshape(xs.shape)

    def contains_grid(self, xs, ys):
        """Boolean mask: points inside or ON any footprint (boundary-inclusive,
        matching the scalar covers() semantics). Bulk STRtree query."""
        import numpy as np
        mask = np.zeros(xs.shape, dtype=bool)
        if not self.available:
            return mask
        import shapely
        pts = shapely.points(xs.ravel(), ys.ravel())
        # covered_by is the boundary-inclusive twin of covers(); one bulk query
        # replaces a Python loop over polygons with prepared contains_xy.
        res = self._tree.query(pts, predicate="covered_by")
        if len(res[0]) == 0:
            return mask
        flat = mask.ravel()
        flat[res[0]] = True
        return flat.reshape(xs.shape)


def building_crossings_db(n_crossings: int) -> float:
    """Wall-penetration loss for a segment crossing n buildings."""
    if n_crossings <= 0:
        return 0.0
    loss = FIRST_CROSSING_DB + (n_crossings - 1) * EXTRA_CROSSING_DB
    return min(loss, MAX_CROSSING_DB)


# ──────────────────────────────────────────────────────────────────────────────
# Process-wide caches: the DEM raster and building STRtree are expensive to
# open/build and are reused across Run clicks.
# ──────────────────────────────────────────────────────────────────────────────

_TERRAIN_SAMPLER_SINGLETON = None
_BUILDING_INDEX_CACHE = {"gdf_ref": None, "index": None}


def get_terrain_sampler() -> "_TerrainSampler":
    global _TERRAIN_SAMPLER_SINGLETON
    if _TERRAIN_SAMPLER_SINGLETON is None:
        _TERRAIN_SAMPLER_SINGLETON = _TerrainSampler()
    return _TERRAIN_SAMPLER_SINGLETON


def get_building_index(buildings_gdf) -> "_BuildingIndex":
    """Reuse the STRtree when handed the same GeoDataFrame object again."""
    if buildings_gdf is None:
        return _BuildingIndex(None)
    cache = _BUILDING_INDEX_CACHE
    if cache["gdf_ref"] is buildings_gdf and cache["index"] is not None:
        return cache["index"]
    idx = _BuildingIndex(buildings_gdf)
    cache["gdf_ref"] = buildings_gdf   # strong ref keeps id() stable
    cache["index"] = idx
    return idx


# ──────────────────────────────────────────────────────────────────────────────
# Terrain diffraction
# ──────────────────────────────────────────────────────────────────────────────

def knife_edge_loss(v: float) -> float:
    """ITU-style knife-edge diffraction approximation (continuous in v).

    L(v) = 6.9 + 20*log10(sqrt((v-0.1)^2 + 1) + (v - 0.1))   for v > -0.78
    ~0 dB below that. Continuous at the boundary (~0 dB at v = -0.78) and
    yields the classic ~6 dB grazing loss at v = 0.
    """
    if v <= -0.78:
        return 0.0
    q = v - 0.1
    return min(6.9 + 20.0 * math.log10(math.sqrt(q * q + 1.0) + q), 30.0)


def terrain_blockage_penalty(
    profile_m: List[float],
    distance_km: float,
    frequency_mhz: float,
    tx_height_m: float,
    rx_height_m: float,
    earth_radius_factor: float = 4.0 / 3.0,
) -> float:
    """
    LOS-blockage loss over a terrain profile with 4/3-earth bulge.

    Computes Fresnel-Kirchoff parameter v at the worst obstruction and returns
    the knife-edge loss. Flat/clear profiles yield 0 dB.

    Standard engineering exclusions:
      - Samples within the first/last 10% of the path are ignored: antennas
        sit above their local ground clutter, and the Fresnel parameter
        diverges as d1 or d2 -> 0, producing phantom near-field "blockage".
      - Endpoint distances used in v are floored at 100 m for the same reason.
    """
    n = len(profile_m) - 1
    if n < 2 or distance_km <= 0:
        return 0.0

    # Wavelength in meters: lambda(m) = 299.792458 / f(MHz)
    wavelength_m = 299.792458 / frequency_mhz

    # Absolute endpoint altitudes (ground + antenna)
    tx_alt = profile_m[0] + tx_height_m
    rx_alt = profile_m[-1] + rx_height_m

    worst_deficit = 0.0
    worst_i = 0
    for i in range(1, n):
        t = i / n
        if t < 0.1 or t > 0.9:
            continue  # near-field exclusion band
        d1_km = distance_km * t
        d2_km = distance_km * (1.0 - t)
        # Earth bulge in meters between the endpoints
        bulge_m = (d1_km * d2_km * 1000.0) / (2.0 * EARTH_RADIUS_KM * earth_radius_factor)
        los_alt = tx_alt * (1.0 - t) + rx_alt * t
        deficit = profile_m[i] + bulge_m - los_alt
        if deficit > worst_deficit:
            worst_deficit = deficit
            worst_i = i

    if worst_deficit <= 0.0:
        return 0.0

    t = worst_i / n
    d1_m = max(distance_km * t * 1000.0, 100.0)
    d2_m = max(distance_km * (1.0 - t) * 1000.0, 100.0)
    v = worst_deficit * math.sqrt(2.0 * (d1_m + d2_m) / (wavelength_m * d1_m * d2_m))
    return knife_edge_loss(v)


# ──────────────────────────────────────────────────────────────────────────────
# Core path-loss assembly
# ──────────────────────────────────────────────────────────────────────────────

def _calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Equirectangular distance in kilometers (valid for small areas)."""
    dlat_km = (lat2 - lat1) * 111.32
    dlng_km = (lng2 - lng1) * 111.32 * math.cos(math.radians(lat1))
    return math.sqrt(dlat_km**2 + dlng_km**2)


def _base_geometry_loss(model: str, frequency_mhz: float, distance_km: float,
                        antenna_height_m: float, rx_height_m: float,
                        sampler: _TerrainSampler, ant_lat: float, ant_lng: float,
                        point_lat: float, point_lng: float,
                        **kwargs) -> float:
    """Geometry/terrain part of the link budget (no weather, no buildings)."""
    if distance_km < 0.01:  # far-field floor: 10 m
        distance_km = 0.01

    if model == "ci":
        reference_distance = kwargs.get('ci_reference_distance_m', 1.0)
        path_loss_exponent = kwargs.get('ci_path_loss_exponent', 2.0)
        return close_in_path_loss(frequency_mhz, distance_km,
                                  reference_distance_m=reference_distance,
                                  path_loss_exponent=path_loss_exponent)

    if model == "itm":
        # Longley-Rice validity floor: the ITS algorithm assumes >= ~1 km
        # paths; below that qlrpfl/lrprop produce erratic excess values.
        # Sub-kilometre links use FSPL plus our own profile-based knife-edge,
        # which stays smooth and physical.
        if distance_km < 1.0:
            loss = free_space_path_loss(frequency_mhz, distance_km)
            if sampler.available:
                profile = sampler.profile(ant_lat, ant_lng, point_lat, point_lng)
                loss += terrain_blockage_penalty(
                    profile, max(distance_km, 0.01), frequency_mhz,
                    kwargs.get('tx_height_m', antenna_height_m),
                    kwargs.get('rx_height_m', rx_height_m),
                )
            return loss

        profile = None
        if sampler.available:
            profile = sampler.profile(ant_lat, ant_lng, point_lat, point_lng)
        return itm_path_loss(
            frequency_mhz, distance_km,
            tx_height_m=kwargs.get('tx_height_m', antenna_height_m),
            rx_height_m=kwargs.get('rx_height_m', rx_height_m),
            terrain_type=kwargs.get('terrain_type', "average"),
            surface_refractivity=kwargs.get('surface_refractivity', 301),
            effective_earth_radius_factor=kwargs.get('effective_earth_radius_factor', 4/3),
            ground_permittivity=kwargs.get('ground_permittivity', 15),
            ground_conductivity=kwargs.get('ground_conductivity', 0.005),
            terrain_profile_m=profile,
        )

    if model == "sionna":
        # Simplified urban excess over FSPL until GPU ray tracing is wired in
        fspl = free_space_path_loss(frequency_mhz, distance_km)
        return fspl + min(20.0, distance_km * 2.0)

    # fspl and the geometry base for weather models
    return free_space_path_loss(frequency_mhz, distance_km)


def _weather_attenuation(model: str, frequency_mhz: float, distance_km: float,
                         **kwargs) -> float:
    """Atmospheric/weather term in dB (rain/gas/fog); 0 otherwise."""
    freq_ghz = frequency_mhz / 1000.0
    if distance_km <= 0:
        return 0.0
    if model == "rain":
        rain_rate = kwargs.get('rain_rate_mmh', 0.0)
        k = kwargs.get('rain_k')
        alpha = kwargs.get('rain_alpha')
        return rain_attenuation(freq_ghz, distance_km, rain_rate, k=k, alpha=alpha)
    if model == "gas":
        return gas_attenuation(freq_ghz, distance_km,
                               temperature_c=kwargs.get('temperature_c', 15.0),
                               pressure_hpa=kwargs.get('pressure_hpa', 1013.25),
                               relative_humidity=kwargs.get('relative_humidity', 50.0))
    if model == "fog":
        return fog_attenuation(freq_ghz, distance_km,
                               kwargs.get('fog_liquid_water_density_gm3', 0.05))
    return 0.0


def _link_rssi(ant: Dict[str, Any], lat: float, lng: float, dist_km: float,
               model: str, combining_ctx: Dict[str, Any]) -> float:
    """Full link budget TX->(lat,lng): base + terrain + buildings [+ weather if outdoor]."""
    sampler = combining_ctx["sampler"]
    buildings = combining_ctx["buildings"]
    kwargs = dict(combining_ctx["kwargs"])   # private copy: we pop below

    antenna_height_m = ant.get("height_m", 30.0)
    # Pop so the **kwargs fan-out below can't collide with the positional
    # rx_height_m (audit M-8 follow-up: per-receiver height overrides).
    rx_height_m = float(kwargs.pop('rx_height_m', 1.5))

    # 1. Base geometry loss (with real terrain profile for ITM)
    loss = _base_geometry_loss(model, ant["frequency_mhz"], dist_km,
                               antenna_height_m, rx_height_m, sampler,
                               ant["lat"], ant["lng"], lat, lng, **kwargs)

    # 2. Weather attenuation - outdoor points only (user directive)
    if model in WEATHER_MODELS:
        indoor = buildings.contains(lat, lng) if buildings.available else False
        if not indoor:
            loss += _weather_attenuation(model, ant["frequency_mhz"], dist_km, **kwargs)

    # 3. Terrain blockage for non-ITM geometry models (ITM handles profiles natively)
    if model != "itm" and sampler.available:
        profile = sampler.profile(ant["lat"], ant["lng"], lat, lng)
        loss += terrain_blockage_penalty(profile, dist_km, ant["frequency_mhz"],
                                         antenna_height_m, rx_height_m)

    # 4. Buildings always apply
    if buildings.available:
        n_cross = buildings.crossings(ant["lat"], ant["lng"], lat, lng)
        loss += building_crossings_db(n_cross)
        if buildings.contains(lat, lng):
            loss += INDOOR_PENALTY_DB

    return ant["tx_power_dbm"] + ant["gain_dbi"] - loss


def combine_rssi(rssi_values: List[float], combining: str) -> float:
    """Combine per-transmitter RSSI values.

    superposition: power-domain sum (physics of incoherent signals)
    best_server:   simple maximum (handover-style)
    """
    if not rssi_values:
        return -200.0
    if combining == "best_server":
        return max(rssi_values)
    import numpy as np
    linear_sum = sum(10.0 ** (r / 10.0) for r in rssi_values)
    return 10.0 * math.log10(linear_sum) if linear_sum > 0 else -200.0


def _zone_from_rssi(rssi: float):
    if rssi >= RSSI_GOOD:
        return 2, "good", ZONE_COLORS["good"]
    if rssi >= RSSI_MEDIUM:
        return 1, "medium", ZONE_COLORS["medium"]
    return 0, "bad", ZONE_COLORS["bad"]


# ──────────────────────────────────────────────────────────────────────────────
# Vectorized grid pipeline (NumPy/shapely bulk ops)
#
# Every function below reproduces its scalar counterpart EXACTLY (same guards,
# same formulas, same tie-breaking); equivalence is enforced by
# tests/propagation/test_coverage_physics.py::TestVectorEquivalence.
# ──────────────────────────────────────────────────────────────────────────────

def _knife_edge_vec(v):
    """Vector twin of knife_edge_loss() (ITU continuous Lee variant)."""
    import numpy as np
    q = v - 0.1
    L = 6.9 + 20.0 * np.log10(np.sqrt(q * q + 1.0) + q)
    L = np.where(v <= -0.78, 0.0, L)
    return np.minimum(L, 30.0)


def _terrain_penalty_grid(ELEV, dist_km, frequency_mhz, tx_height_m, rx_height_m,
                          earth_radius_factor=4.0 / 3.0):
    """Vector twin of terrain_blockage_penalty() over a whole link matrix.

    ELEV: (H, W, S) ground-elevation profiles per link (endpoints included).
    """
    import numpy as np
    H, W, S = ELEV.shape
    out = np.zeros((H, W))
    if S < 3:
        return out

    tx_alt = ELEV[..., 0] + tx_height_m
    rx_alt = ELEV[..., -1] + rx_height_m

    best_def = np.zeros((H, W))
    best_t = np.zeros((H, W))
    for k in range(1, S):
        t = k / (S - 1)
        if t < 0.1 or t > 0.9:
            continue  # near-field exclusion band (matches scalar)
        d1 = dist_km * t
        d2 = dist_km * (1.0 - t)
        bulge = (d1 * d2 * 1000.0) / (2.0 * EARTH_RADIUS_KM * earth_radius_factor)
        los = tx_alt * (1.0 - t) + rx_alt * t
        deficit = ELEV[..., k] + bulge - los
        upd = deficit > best_def          # strict '>' preserves first-max ties
        best_def = np.where(upd, deficit, best_def)
        best_t = np.where(upd, t, best_t)

    has = best_def > 0
    if not has.any():
        return out
    d1_m = np.maximum(dist_km * best_t * 1000.0, 100.0)
    d2_m = np.maximum(dist_km * (1.0 - best_t) * 1000.0, 100.0)
    lam = 299.792458 / frequency_mhz
    v = best_def * np.sqrt(2.0 * (d1_m + d2_m) / (lam * d1_m * d2_m))
    out[...] = np.where(has, _knife_edge_vec(v), 0.0)
    return out


def _weather_matrix(model, freq_ghz, dist_km, **kw):
    """Element-exact vector twin of _weather_attenuation()."""
    import numpy as np
    zero = np.zeros_like(dist_km)
    if model == "rain":
        R = float(kw.get('rain_rate_mmh', 0.0))
        f_ok = freq_ghz > 0
        valid = f_ok & (dist_km > 0) & (R > 0)
        k = kw.get('rain_k')
        alpha = kw.get('rain_alpha')
        if k is None and alpha is None:
            k = 0.0001 * np.power(np.maximum(freq_ghz, 1e-12), 0.88)
            alpha = 0.90
        elif k is None:
            k = 0.0001 * np.power(np.maximum(freq_ghz, 1e-12), 0.88)
        elif alpha is None:
            alpha = 0.90
        gamma = k * np.power(R, alpha)
        return np.where(valid & (gamma > 0), gamma * np.maximum(dist_km, 0.0), zero)
    if model == "gas":
        temp = float(kw.get('temperature_c', 15.0))
        press = float(kw.get('pressure_hpa', 1013.25))
        rh = float(kw.get('relative_humidity', 50.0))
        tk = temp + 273.15
        svp = 6.1121 * np.exp((17.502 * temp) / (temp + 240.97))
        wvd = 216.7 * (rh / 100.0 * svp) / tk
        g_o = 0.0001 * press * freq_ghz ** 2 / (freq_ghz ** 2 + 0.1)
        g_w = 0.000045 * wvd * freq_ghz ** 2 / (freq_ghz ** 2 + 0.5)
        return np.where(dist_km > 0, (g_o + g_w) * dist_km, zero)
    if model == "fog":
        M = float(kw.get('fog_liquid_water_density_gm3', 0.05))
        valid = (freq_ghz > 0) & (dist_km > 0) & (M > 0)
        gamma = 0.2 * M * freq_ghz ** 2 / (freq_ghz ** 2 + 0.7)
        return np.where(valid, gamma * dist_km, zero)
    return zero


def _base_loss_matrix(model, freq_mhz, dist_km, ref_distance_m, path_loss_exp):
    """Vector twin of the geometry base losses (fspl/ci/sionna fallback)."""
    import numpy as np
    d = np.maximum(dist_km, 0.01)                       # far-field floor
    if model == "ci":
        d0_m = np.maximum(float(ref_distance_m), 1e-6)
        d_m = np.maximum(d * 1e3, d0_m)
        return 32.44 + 20.0 * np.log10(np.maximum(freq_mhz, 1e-9)) \
            + 20.0 * np.log10(d0_m / 1000.0) \
            + 10.0 * path_loss_exp * np.log10(d_m / d0_m)
    if model == "sionna":
        return 32.44 + 20.0 * np.log10(np.maximum(freq_mhz, 1e-9)) \
            + 20.0 * np.log10(d) + np.minimum(20.0, d * 2.0)
    # fspl (+ geometry base of weather models)
    return 32.44 + 20.0 * np.log10(np.maximum(freq_mhz, 1e-9)) \
        + 20.0 * np.log10(d)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def box_bounds(center_lat: float, center_lng: float, box_size_m: float):
    """Return (min_lat, min_lng, max_lat, max_lng) for a square analysis area
    of side `box_size_m` centered on (center_lat, center_lng)."""
    half_lat = (box_size_m / 2.0) / (111.32 * 1000.0)
    half_lng = (box_size_m / 2.0) / (111.32 * 1000.0 * math.cos(math.radians(center_lat)))
    return (center_lat - half_lat, center_lng - half_lng,
            center_lat + half_lat, center_lng + half_lng)


def buildings_fetch_needed(meta, center_lat: float, center_lng: float,
                           fetch_dist: float) -> bool:
    """Audit M-6: True when cached OSM footprints no longer describe the
    analysis center — never fetched (`meta is None`) or the center moved
    beyond half the original fetch radius."""
    if meta is None:
        return True
    dlat_m = (center_lat - meta["lat"]) * 111_320.0
    dlng_m = ((center_lng - meta["lng"]) * 111_320.0
              * math.cos(math.radians(meta["lat"])))
    return math.hypot(dlat_m, dlng_m) > 0.5 * fetch_dist


def compute_coverage_grid(antennas: List[Dict[str, Any]], center_lat: float, center_lng: float,
                          box_size_m: float, resolution_m: float, model: str = "fspl",
                          combining: str = "superposition",
                          buildings_gdf=None,
                          engine: str = "vector",
                          **propagation_kwargs) -> List[Dict[str, Any]]:
    """
    Compute RF coverage over a user-defined square analysis area.

    The analysis region is exactly the box given by (center_lat, center_lng,
    box_size_m). Every transmitter contributes to EVERY point inside the box —
    there is no per-transmitter range limit and no sentinel values: signal
    decays continuously with distance via the propagation model itself, so
    distant areas simply fall into worse zones.

    Args:
        antennas: Antenna dicts (lat, lng, name, frequency_mhz, tx_power_dbm,
                  gain_dbi, height_m, nature). Only transmitters radiate.
        center_lat/lng: Center of the analysis box (typically the map center).
        box_size_m: Side length of the square analysis area in meters.
        resolution_m: Grid spacing in meters.
        model: Base propagation model. Weather selections (rain/gas/fog)
               automatically include an FSPL base + the weather term.
        combining: "superposition" (default) | "best_server".
        buildings_gdf: Optional GeoDataFrame of OSM footprints used for
                       wall-penetration losses.
        engine: "vector" (default; NumPy/shapely bulk ops) | "scalar"
                (reference implementation; kept for validation).
        **propagation_kwargs: Weather/model parameters (rain_rate_mmh, ...).

    Returns list of dicts covering the full box grid:
        lat, lng, rssi_dbm, zone, color, best_antenna.
    """
    if not antennas or box_size_m <= 0:
        return []

    model = (model or "fspl").lower()
    transmitters = [a for a in antennas if a.get("nature", "transmitter") == "transmitter"]
    if not transmitters:
        return []
    if resolution_m <= 0:
        resolution_m = 25.0

    n_side = int(round(box_size_m / resolution_m)) + 1
    total_links = n_side * n_side * len(transmitters)

    # ITM runs a full Longley-Rice computation per TX x point link; warn when
    # the requested workload is heavy so users can pick FSPL/CI or coarsen.
    if model == "itm" and total_links > 5000:
        print(f"WARNING: ITM over {box_size_m:.0f} m box at {resolution_m:.0f} m "
              f"resolution with {len(transmitters)} transmitter(s) requires "
              f"~{total_links} Longley-Rice runs; this may take minutes. "
              f"Consider fspl/ci or a coarser resolution.")
    elif total_links > 40000:
        print(f"NOTE: {box_size_m:.0f} m box at {resolution_m:.0f} m resolution "
              f"with {len(transmitters)} transmitter(s) = ~{total_links} links. "
              f"A coarser resolution will run faster.")

    if engine == "scalar":
        return _grid_scalar(transmitters, center_lat, center_lng, box_size_m,
                            resolution_m, model, combining, buildings_gdf,
                            propagation_kwargs)
    return _grid_vector(transmitters, center_lat, center_lng, box_size_m,
                        resolution_m, model, combining, buildings_gdf,
                        propagation_kwargs)


def compute_coverage_result(antennas: List[Dict[str, Any]], center_lat: float,
                            center_lng: float, box_size_m: float,
                            resolution_m: float, model: str = "fspl",
                            combining: str = "superposition",
                            buildings_gdf=None,
                            **propagation_kwargs) -> Optional[Dict[str, Any]]:
    """Vector-engine coverage with matrix extras for the UI's raster renderer.

    Returns a dict with:
      points      — same list-of-dicts as compute_coverage_grid
      rssi        — (H, W) float matrix of combined RSSI (dBm), lat-major
      zone_code   — (H, W) int8: 0 bad, 1 medium, 2 good
      lats/lngs   — 1D sample coordinate arrays
      bounds      — (min_lat, min_lng, max_lat, max_lng) of the analysis box
      stats       — zone_statistics(points)
      points_md5  — md5 of the canonical points JSON (O(1) fingerprint input)

    Returns None when there are no transmitters to evaluate.
    """
    transmitters = [a for a in antennas or []
                    if a.get("nature", "transmitter") == "transmitter"]
    if not antennas or box_size_m <= 0 or not transmitters:
        return None

    model = (model or "fspl").lower()
    if resolution_m <= 0:
        resolution_m = 25.0
    # Audit L-17: the result path needs the same workload advisories as
    # compute_coverage_grid (ITM is a full qlrpfl per link; anything huge is
    # going to take a while).
    n_side = int(round(box_size_m / resolution_m)) + 1
    total_links = n_side * n_side * len(transmitters)
    if model == "itm" and total_links > 5000:
        print(f"WARNING: ITM over {box_size_m:.0f} m box at {resolution_m:.0f} m "
              f"resolution with {len(transmitters)} transmitter(s) requires "
              f"~{total_links} Longley-Rice runs; this may take minutes.")
    elif total_links > 40000:
        print(f"NOTE: ~{total_links} links requested; consider a coarser "
              f"resolution for faster runs.")
    points, extras = _grid_vector(transmitters, center_lat, center_lng,
                                  box_size_m, resolution_m, model, combining,
                                  buildings_gdf, propagation_kwargs,
                                  want_matrices=True)
    return {
        "points": points,
        "rssi": extras["rssi"],
        "zone_code": extras["zone_code"],
        "lats": extras["lats"],
        "lngs": extras["lngs"],
        "bounds": extras["bounds"],
        "stats": zone_statistics(points),
        "points_md5": extras["points_md5"],
        "warnings": extras.get("warnings", []),   # audit C1: ITM reliability
        # Audit L-7: snapshot the parameters ACTUALLY used for this run so
        # exports can describe their own data even after sidebar edits.
        "run_params": {
            "model": model,
            "combining": combining,
            "box_size_m": float(box_size_m),
            "resolution_m": float(resolution_m),
            "weather": {k: v for k, v in propagation_kwargs.items()
                        if k in ("rain_rate_mmh", "relative_humidity",
                                 "fog_liquid_water_density_gm3", "temperature_c")},
            "buildings_used": buildings_gdf is not None,
        },
    }


def _grid_scalar(transmitters, center_lat, center_lng, box_size_m, resolution_m,
                 model, combining, buildings_gdf, propagation_kwargs):
    """Reference implementation: per-link scalar pipeline. Kept for
    validation against the vectorized engine (see TestVectorEquivalence)."""
    sampler = get_terrain_sampler()
    buildings = get_building_index(buildings_gdf)
    ctx = {"sampler": sampler, "buildings": buildings, "kwargs": dict(propagation_kwargs)}

    min_lat, min_lng, max_lat, max_lng = box_bounds(center_lat, center_lng, box_size_m)
    n_lat = max(int(round(box_size_m / resolution_m)), 1)
    lats = [min_lat + i * (max_lat - min_lat) / n_lat for i in range(n_lat + 1)]
    lngs = [min_lng + j * (max_lng - min_lng) / n_lat for j in range(n_lat + 1)]

    coverage_points = []
    for lat in lats:
        for lng in lngs:
            contributions = []   # (rssi, name)
            for ant in transmitters:
                dist_km = _calculate_distance(ant['lat'], ant['lng'], lat, lng)
                rssi = _link_rssi(ant, lat, lng, dist_km, model, ctx)
                contributions.append((rssi, ant.get('name', 'Unknown')))

            total_rssi = combine_rssi([c[0] for c in contributions], combining)
            best_name = max(contributions, key=lambda c: c[0])[1]
            _, zone_str, color = _zone_from_rssi(total_rssi)

            coverage_points.append({
                "lat": lat,
                "lng": lng,
                "rssi_dbm": round(total_rssi, 2),
                "zone": zone_str,
                "color": color,
                "best_antenna": best_name,
            })

    return coverage_points


def _grid_vector(transmitters, center_lat, center_lng, box_size_m, resolution_m,
                 model, combining, buildings_gdf, kw_in, want_matrices=False):
    """Vectorized engine: NumPy matrices + shapely bulk predicates.

    Produces bit-for-bit comparable results to _grid_scalar (validated by
    TestVectorEquivalence) while eliminating per-link interpreted Python.

    When `want_matrices` is True, returns (coverage_points, extras) where
    extras carries the raw RSSI/zone matrices and grid metadata for the UI's
    raster renderer.
    """
    import numpy as np

    kwargs = dict(kw_in)
    sampler = get_terrain_sampler()
    buildings = get_building_index(buildings_gdf)

    min_lat, min_lng, max_lat, max_lng = box_bounds(center_lat, center_lng, box_size_m)
    n_side = max(int(round(box_size_m / resolution_m)), 1)
    lats = np.linspace(min_lat, max_lat, n_side + 1)
    lngs = np.linspace(min_lng, max_lng, n_side + 1)
    LAT, LNG = np.meshgrid(lats, lngs, indexing="ij")
    H, W = LAT.shape

    # TX-independent indoor mask — computed ONCE for the whole grid.
    # NOTE coordinate order: shapely x=lng, y=lat (swapping these silently
    # disables all building interactions — regression-tested).
    indoor_mask = buildings.contains_grid(LNG, LAT) if buildings.available else None

    rx_height_m = float(kwargs.get('rx_height_m', 1.5))
    ref_d = float(kwargs.get('ci_reference_distance_m', 1.0))
    p_exp = float(kwargs.get('ci_path_loss_exponent', 2.0))

    rssi_stack = []
    names = []
    # ITM reliability diagnostics (audit C1): snapshot the wrapper's module
    # hooks before the loop; anything that changed during it becomes a
    # user-facing warning on the result.
    import propagation_model.itm_model as _itmm
    _diag_before = (_itmm._LAST_FALLBACK_REASON, _itmm._LAST_KWX_WARNING)
    for ant in transmitters:
        freq = float(ant["frequency_mhz"])
        tx_h = float(ant.get("height_m", 30.0))

        dlat_km = (LAT - ant['lat']) * 111.32
        dlng_km = (LNG - ant['lng']) * 111.32 * math.cos(math.radians(ant['lat']))
        dist_km = np.sqrt(dlat_km ** 2 + dlng_km ** 2)

        # 1. Base geometry loss
        loss = _base_loss_matrix(model, freq, dist_km, ref_d, p_exp)

        # ITM: >=1 km links go through real Longley-Rice (validity floor);
        # sub-km cells keep FSPL+knife-edge from above. The >=1 km loop runs
        # EVEN WITHOUT a DEM (flat synthetic profile) so the vector engine
        # matches the scalar engine's behaviour exactly — previously the two
        # engines diverged silently on DEM-less machines.
        ELEV = None
        use_terrain = False
        if model == "itm":
            if sampler.available:
                ELEV = sampler.sample_grid(ant['lat'], ant['lng'], LAT, LNG)
                terr = _terrain_penalty_grid(ELEV, dist_km, freq, tx_h, rx_height_m)
                loss = loss + terr
            big = dist_km >= 1.0
            if big.any():
                terrain_type = kwargs.get('terrain_type', "average")
                surf_refr = kwargs.get('surface_refractivity', 301)
                eerf = kwargs.get('effective_earth_radius_factor', 4 / 3)
                g_perm = kwargs.get('ground_permittivity', 15)
                g_cond = kwargs.get('ground_conductivity', 0.005)
                for i, j in zip(*np.where(big)):
                    prof = ELEV[i, j, :].tolist() if ELEV is not None else None
                    loss[i, j] = itm_path_loss(
                        freq, float(dist_km[i, j]),
                        tx_height_m=tx_h, rx_height_m=rx_height_m,
                        terrain_type=terrain_type,
                        surface_refractivity=surf_refr,
                        effective_earth_radius_factor=eerf,
                        ground_permittivity=g_perm,
                        ground_conductivity=g_cond,
                        terrain_profile_m=prof,
                    )
        else:
            use_terrain = sampler.available

        # 2. Weather attenuation — outdoor points only
        if model in WEATHER_MODELS:
            outdoor = ~(indoor_mask.astype(bool)) if indoor_mask is not None \
                else np.ones((H, W), dtype=bool)
            loss = loss + np.where(outdoor,
                                   _weather_matrix(model, freq / 1000.0, dist_km, **kwargs),
                                   0.0)

        # 3. Terrain diffraction for non-ITM geometry models
        if use_terrain and model != "itm":
            if ELEV is None:
                ELEV = sampler.sample_grid(ant['lat'], ant['lng'], LAT, LNG)
            loss = loss + _terrain_penalty_grid(ELEV, dist_km, freq, tx_h, rx_height_m)

        # 4. Buildings always apply
        if buildings.available:
            counts = buildings.crossings_grid(ant['lat'], ant['lng'], LNG, LAT).astype("float64")
            cross_db = np.where(counts > 0,
                                np.minimum(FIRST_CROSSING_DB
                                           + (np.maximum(counts, 1) - 1) * EXTRA_CROSSING_DB,
                                           MAX_CROSSING_DB),
                                0.0)
            loss = loss + cross_db
            if indoor_mask is not None:
                loss = loss + np.where(indoor_mask, INDOOR_PENALTY_DB, 0.0)

        rssi_stack.append(ant["tx_power_dbm"] + ant["gain_dbi"] - loss)
        names.append(ant.get("name", "Unknown"))

    stack = np.stack(rssi_stack)                     # (T, H, W)
    linear_sum = np.power(10.0, stack / 10.0).sum(axis=0)
    # Underflow parity with the scalar engine: an exactly-zero power sum is
    # reported as the -200 dBm sentinel, never as a log-of-tiny absurdity.
    total = np.where(linear_sum > 0,
                     10.0 * np.log10(np.maximum(linear_sum, 1e-300)),
                     -200.0)
    argbest = np.argmax(stack, axis=0)
    if combining != "superposition":
        total = stack.max(axis=0)

    zone_codes = np.zeros(total.shape, dtype="int8")          # 0 bad, 1 mid, 2 good
    zone_codes[total >= RSSI_MEDIUM] = 1
    zone_codes[total >= RSSI_GOOD] = 2

    extras = None
    if want_matrices:
        import hashlib as _hashlib
        # Aggregate ITM reliability warnings (audit C1): only report what THIS
        # run produced, so stale diagnostics from earlier runs don't linger.
        warns = []
        reason_now, kwx_now = (_itmm._LAST_FALLBACK_REASON,
                               _itmm._LAST_KWX_WARNING)
        if reason_now and reason_now != _diag_before[0]:
            warns.append(f"ITM fell back to approximate model ({reason_now})")
        if kwx_now and kwx_now != _diag_before[1]:
            warns.append(kwx_now)
        extras = {
            "rssi": total,
            "zone_code": zone_codes,
            "lats": lats,
            "lngs": lngs,
            "bounds": (min_lat, min_lng, max_lat, max_lng),
            "warnings": warns,
        }

    # Assemble output (row-major lat asc, lng asc — matches scalar order)
    coverage_points = []
    flat_total = total.ravel()
    flat_best = argbest.ravel()
    k = 0
    for i in range(H):
        for j in range(W):
            trssi = float(flat_total[k])
            _, zone_str, color = _zone_from_rssi(trssi)
            coverage_points.append({
                "lat": float(lats[i]),
                "lng": float(lngs[j]),
                "rssi_dbm": round(trssi, 2),
                "zone": zone_str,
                "color": color,
                "best_antenna": names[int(flat_best[k])],
            })
            k += 1

    if want_matrices:
        payload = json.dumps(coverage_points, sort_keys=True,
                             default=str).encode("utf-8")
        extras["points_md5"] = _hashlib.md5(payload).hexdigest()
        return coverage_points, extras
    return coverage_points


def evaluate_receivers(receivers: List[Dict[str, Any]], transmitters: List[Dict[str, Any]],
                       model: str = "fspl",
                       combining: str = "superposition",
                       buildings_gdf=None,
                       **propagation_kwargs) -> List[Dict[str, Any]]:
    """
    Compute the combined received signal at each receiver location from ALL
    transmitters (no range limit).

    Returns list of dicts: name, lat, lng, rssi_dbm, zone, color,
    serving_antenna (strongest contributor), covered (True when RSSI reaches
    at least the medium-zone threshold).
    """
    if not receivers or not transmitters:
        return []

    model = (model or "fspl").lower()
    sampler = get_terrain_sampler()
    buildings = get_building_index(buildings_gdf)

    results = []
    for rx in receivers:
        # Per-receiver antenna height (audit M-8): a receiver's own height_m
        # overrides the global rx_height_m for its link budget.
        rx_kwargs = dict(propagation_kwargs)
        if rx.get("height_m") is not None:
            rx_kwargs["rx_height_m"] = float(rx["height_m"])
        rx_ctx = {"sampler": sampler, "buildings": buildings, "kwargs": rx_kwargs}

        contributions = []
        for ant in transmitters:
            dist_km = _calculate_distance(ant['lat'], ant['lng'], rx['lat'], rx['lng'])
            contributions.append((_link_rssi(ant, rx['lat'], rx['lng'], dist_km,
                                             model, rx_ctx),
                                  ant.get('name', 'Unknown')))

        total = combine_rssi([c[0] for c in contributions], combining)
        serving = max(contributions, key=lambda c: c[0])[1]
        _, zone_str, color = _zone_from_rssi(total)
        covered = total >= RSSI_MEDIUM

        results.append({
            "name": rx.get("name", "Receiver"),
            "lat": rx["lat"],
            "lng": rx["lng"],
            "rssi_dbm": round(total, 2),
            "zone": zone_str,
            "color": color,
            "serving_antenna": serving,
            "covered": bool(covered),
        })
    return results


def zone_statistics(coverage_points: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate zone counts/percentages plus average RSSI."""
    if not coverage_points:
        return {
            "total": 0, "good": 0, "medium": 0, "bad": 0,
            "good_pct": 0, "medium_pct": 0, "bad_pct": 0,
            "avg_rssi_dbm": 0.0
        }

    total = len(coverage_points)
    good = sum(1 for p in coverage_points if p["zone"] == "good")
    medium = sum(1 for p in coverage_points if p["zone"] == "medium")
    bad = sum(1 for p in coverage_points if p["zone"] == "bad")

    good_pct = round((good / total) * 100, 1) if total > 0 else 0.0
    medium_pct = round((medium / total) * 100, 1) if total > 0 else 0.0
    bad_pct = round((bad / total) * 100, 1) if total > 0 else 0.0

    avg_rssi = sum(p["rssi_dbm"] for p in coverage_points) / total

    return {
        "total": total,
        "good": good,
        "medium": medium,
        "bad": bad,
        "good_pct": good_pct,
        "medium_pct": medium_pct,
        "bad_pct": bad_pct,
        "avg_rssi_dbm": round(avg_rssi, 2)
    }
