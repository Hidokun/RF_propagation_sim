"""
input_data_collection/ingestion.py

Parses environmental / scenario inputs from various file formats (CSV, JSON, XML, KML)
or from interactive terminal prompts. Normalizes everything into a standard
SimulationConfig dataclass consumed by the RF_prop_sim engine (main.py).
"""

import os
import json
import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from typing import Optional, List

# ─── Standard Simulation Configuration ─────────────────────────────────────────

@dataclass
class ReceiverPoint:
    lat: float
    lng: float
    height_m: float = 1.5   # typical UE height

@dataclass
class SimulationConfig:
    """
    Full simulation scenario configuration consumed by RF_prop_sim/main.py.
    """
    # Location
    address:        str   = "casablanca, Morocco"
    center_lat:     Optional[float] = None
    center_lng:     Optional[float] = None
    area_radius_m:  float = 500.0           # radius for building extraction (meters)

    # Propagation
    model:          str   = "fspl"          # fspl | rain | ci | itm | sionna
    frequency_mhz:  float = 900.0
    distance_km:    float = 0.5
    tx_height_m:    float = 30.0
    rx_height_m:    float = 1.5
    rain_rate_mmh:  float = 0.0            # mm/h; only used if model == "rain"
    # Rain model optional coefficients (specific attenuation model gamma = k * R^alpha)
    rain_k:         Optional[float] = None
    rain_alpha:     Optional[float] = None
    fog_liquid_water_density_gm3: float = 0.05
    temperature_c:  float = 15.0           # ambient temperature for atmospheric attenuation
    pressure_hpa:   float = 1013.25        # atmospheric pressure for gas attenuation
    relative_humidity: float = 50.0        # relative humidity (%) for gas attenuation
    surface_refractivity: Optional[float] = None  # optional override for ITM surface refractivity
    effective_earth_radius_factor: float = 4.0/3.0
    ground_permittivity: float = 15.0
    ground_conductivity: float = 0.005
    # Close-In (CI) model parameters
    ci_path_loss_exponent: float = 2.0
    ci_reference_distance_m: float = 1.0
    tx_power_dbm: float = 40.0
    antenna_gain_dbi: float = 0.0
    antenna_lat: Optional[float] = None
    antenna_lng: Optional[float] = None
    antenna_alt_m: Optional[float] = None
    antenna_config_path: Optional[str] = None
    terrain_type:   str   = "average"      # for ITM: average | urban | suburban | rural
    # Multi-transmitter overlap physics: power-sum ("superposition") or max ("best_server")
    combining:      str   = "superposition"

    # Optimization
    run_optimization: bool  = False
    opt_area_km:      float = 1.0

    # DEM
    run_dem:        bool = False

    # Output directory for generated artifacts (coverage maps, reports)
    output_dir:     Optional[str] = None

    # Receivers (optional list of specific points)
    receivers:      List[ReceiverPoint] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        return d


# ─── Dispatcher ────────────────────────────────────────────────────────────────

def load_config_from_file(path: str) -> SimulationConfig:
    """Load a simulation config from a file path using the parser entrypoint."""
    return parse_simulation_config(path)


def parse_simulation_config(source=None):
    """
    Main entry point. Accepts:
      - None: returns defaults (used when orchestrator fills programmatically)
      - dict: raw key-value configuration
      - str: path to a file (.csv, .json, .xml, .kml) or text
      - SimulationConfig: passed through unchanged
    """
    if source is None:
        return SimulationConfig()

    if isinstance(source, SimulationConfig):
        return source

    if isinstance(source, dict):
        return _from_dict(source)

    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Input config file not found: {source}")
        ext = os.path.splitext(source)[1].lower()
        parsers = {
            ".json": _from_json,
            ".csv":  _from_csv,
            ".xml":  _from_xml,
            ".kml":  _from_kml,
        }
        parser = parsers.get(ext, _from_generic_text)
        return parser(source)

    raise TypeError(f"Unsupported config source type: {type(source)}")


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _apply_dict_to_config(cfg: SimulationConfig, d: dict) -> SimulationConfig:
    """Apply a flattened dict of values to an existing SimulationConfig."""
    aliases = {
        "freq": "frequency_mhz",
        "frequency": "frequency_mhz",
        "dist": "distance_km",
        "distance": "distance_km",
        "model_name": "model",
        "propagation_model": "model",
        "area": "area_radius_m",
        "radius": "area_radius_m",
        "tx_height": "tx_height_m",
        "rx_height": "rx_height_m",
        "latitude": "center_lat",
        "longitude": "center_lng",
        "rain": "rain_rate_mmh",
        "rain_rate": "rain_rate_mmh",
        # Legacy keys (audit fix #4): previously fed dead fields; now mapped
        # onto the one fog parameter the model actually consumes (g/m³ LWD).
        "fog": "fog_liquid_water_density_gm3",
        "fog_density": "fog_liquid_water_density_gm3",
        "fog_liquid_water_density": "fog_liquid_water_density_gm3",
        "temperature": "temperature_c",
        "pressure": "pressure_hpa",
        "humidity": "relative_humidity",
        "relative_humidity": "relative_humidity",
        "refractivity": "surface_refractivity",
        "effective_earth_radius_factor": "effective_earth_radius_factor",
        "ground_permittivity": "ground_permittivity",
        "ground_conductivity": "ground_conductivity",
        "tx_power": "tx_power_dbm",
        "tx_power_dbm": "tx_power_dbm",
        "antenna_gain": "antenna_gain_dbi",
        "antenna_gain_dbi": "antenna_gain_dbi",
        "antenna_lat": "antenna_lat",
        "antenna_lng": "antenna_lng",
        "antenna_alt": "antenna_alt_m",
        "antenna_alt_m": "antenna_alt_m",
        "antenna_config": "antenna_config_path",
        "antenna_config_path": "antenna_config_path",
        "optimize": "run_optimization",
    }

    bool_fields = {"run_optimization", "run_dem"}
    float_fields = {
        "center_lat", "center_lng", "area_radius_m", "frequency_mhz",
        "distance_km", "tx_height_m", "rx_height_m", "rain_rate_mmh",
        "fog_liquid_water_density_gm3",
        "temperature_c", "pressure_hpa", "relative_humidity",
        "surface_refractivity", "effective_earth_radius_factor",
        "ground_permittivity", "ground_conductivity", "tx_power_dbm",
        "antenna_gain_dbi", "antenna_lat", "antenna_lng", "antenna_alt_m",
        "opt_area_km"
    }

    for k, v in d.items():
        raw_key = k.lower().strip() if isinstance(k, str) else ""
        key = aliases.get(raw_key, raw_key)
        if not key or not hasattr(cfg, key):
            continue
        try:
            if key in bool_fields:
                # Wider truthy set (audit L-5): HTML forms send "on"/"y".
                setattr(cfg, key,
                        str(v).strip().lower() in ("1", "true", "yes", "y", "on"))
            elif key in float_fields:
                setattr(cfg, key, float(v))
            else:
                setattr(cfg, key, v)
        except (ValueError, TypeError) as e:
            print(f"WARNING: Could not set SimulationConfig field '{key}': {e}")
    return cfg


def _validate_simulation_config(cfg: SimulationConfig) -> SimulationConfig:
    """Validate a normalized simulation config and apply safe defaults."""
    if cfg.frequency_mhz <= 0:
        raise ValueError("frequency_mhz must be positive")
    if cfg.distance_km <= 0:
        raise ValueError("distance_km must be positive")
    if cfg.tx_height_m < 0 or cfg.rx_height_m < 0:
        raise ValueError("tx_height_m and rx_height_m must be non-negative")
    if cfg.area_radius_m <= 0:
        cfg.area_radius_m = 500.0
    # Audit L-2: previously unchecked ranges.
    if cfg.rain_rate_mmh < 0:
        raise ValueError("rain_rate_mmh must be non-negative")
    if not (0.0 <= cfg.relative_humidity <= 100.0):
        raise ValueError("relative_humidity is a percent in [0, 100]")
    for name, val in (("center_lat", cfg.center_lat), ("center_lng", cfg.center_lng),
                      ("antenna_lat", cfg.antenna_lat), ("antenna_lng", cfg.antenna_lng)):
        if val is None:
            continue
        if "lat" in name and not (-90.0 <= val <= 90.0):
            raise ValueError(f"{name}={val} outside [-90, 90]")
        if "lng" in name and not (-180.0 <= val <= 180.0):
            raise ValueError(f"{name}={val} outside [-180, 180]")
    if any(r.height_m < 0 for r in cfg.receivers):
        raise ValueError("receiver height_m must be non-negative")
    if cfg.model not in {"fspl", "rain", "ci", "itm", "sionna", "gas", "fog"}:
        print(f"WARNING: unknown model '{cfg.model}'; falling back to 'fspl'.")
        cfg.model = "fspl"
    if cfg.combining not in ("superposition", "best_server"):
        print(f"WARNING: unknown combining mode '{cfg.combining}'; "
              f"falling back to 'superposition'.")
        cfg.combining = "superposition"
    return cfg


def _from_dict(d: dict) -> SimulationConfig:
    cfg = SimulationConfig()
    cfg = _apply_dict_to_config(cfg, d)
    return _validate_simulation_config(cfg)


def _from_json(path: str) -> SimulationConfig:
    with open(path, "r") as f:
        data = json.load(f)
    # Audit L-4: null / [null] JSON must fail loudly, not crash on dict ops.
    if isinstance(data, list):
        data = data[0] if data and isinstance(data[0], dict) else None
    if not isinstance(data, dict):
        raise ValueError(f"JSON config must be an object: {path}")
    print(f"Loaded simulation config from JSON: {path}")
    return _from_dict(data)


def _from_csv(path: str) -> SimulationConfig:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV file is empty: {path}")
    # Support both single-row configs and multi-row receiver grids
    if len(rows) == 1:
        print(f"Loaded simulation config from CSV: {path}")
        return _from_dict(dict(rows[0]))
    else:
        # Receiver grid: requires lat/lng columns (case-insensitive); rows
        # that fail numeric parsing are tallied and reported, never silently
        # dropped (audit M-7).
        lowered = [{k.lower(): v for k, v in row.items()} for row in rows]
        first = lowered[0]
        if "lat" not in first and "latitude" not in first:
            raise ValueError(
                f"Receiver CSV {path}: missing required 'lat'/'latitude' column "
                f"(found: {sorted(first.keys())})"
            )
        # Accept the common 'long' spelling alongside 'lng'/'longitude'.
        has_lng = any(k in first for k in ("lng", "longitude", "long"))
        if not has_lng:
            raise ValueError(
                f"Receiver CSV {path}: missing required 'lng'/'longitude'/'long' column "
                f"(found: {sorted(first.keys())})"
            )
        cfg = SimulationConfig()
        receivers = []
        dropped = []
        for i, row in enumerate(lowered, start=1):
            try:
                lat_raw = row.get("lat", row.get("latitude"))
                lng_raw = row.get("lng", row.get("longitude", row.get("long")))
                if lat_raw in (None, "") or lng_raw in (None, ""):
                    raise ValueError("empty coordinate cell")
                rx = ReceiverPoint(
                    lat=float(lat_raw),
                    lng=float(lng_raw),
                    height_m=float(row.get("height_m") or 1.5),
                )
                receivers.append(rx)
            except (ValueError, TypeError) as exc:
                dropped.append((i, str(exc)))
        if dropped:
            print(f"WARNING: receiver CSV {path}: dropped {len(dropped)} malformed "
                  f"row(s): {dropped[:5]}{' ...' if len(dropped) > 5 else ''}")
        cfg.receivers = receivers
        print(f"Loaded {len(receivers)} receiver points from CSV: {path}")
        return cfg


def _from_xml(path: str) -> SimulationConfig:
    tree = ET.parse(path)
    root = tree.getroot()
    data = {}
    for child in root:
        # Audit L-3: self-closing elements have text=None -> skip.
        if child.text is not None:
            data[child.tag] = child.text
    print(f"Loaded simulation config from XML: {path}")
    return _from_dict(data)


def _from_kml(path: str) -> SimulationConfig:
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    tree = ET.parse(path)
    root = tree.getroot()
    data = {}

    coords_el = root.find(".//kml:coordinates", ns)
    if coords_el is not None and coords_el.text:
        parts = coords_el.text.strip().split(",")
        if len(parts) >= 2:
            data["center_lng"] = parts[0].strip()
            data["center_lat"] = parts[1].strip()

    for sd in root.findall(".//kml:SimpleData", ns):
        if sd.text is not None:                      # audit L-3
            data[sd.get("name", "")] = sd.text

    print(f"Loaded simulation config from KML: {path}")
    return _from_dict(data)


def _from_generic_text(path: str) -> SimulationConfig:
    data = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for sep in ["=", ":"]:
                if sep in line:
                    k, _, v = line.partition(sep)
                    data[k.strip()] = v.strip()
                    break
    print(f"Loaded simulation config from text file: {path}")
    return _from_dict(data)
