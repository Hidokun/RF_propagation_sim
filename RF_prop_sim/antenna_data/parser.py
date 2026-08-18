"""
antenna_data/parser.py

Parses antenna configuration from various file formats (CSV, JSON, XML, KML)
or from a direct dictionary. Normalizes everything into a standard AntennaConfig
dataclass that the propagation models can consume.
"""

import os
import json
import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from typing import Optional

# ─── Standard Antenna Configuration ────────────────────────────────────────────

@dataclass
class AntennaConfig:
    """
    Normalized antenna specification consumed by all propagation models.
    All units are SI or as explicitly noted.
    """
    name:           str   = "antenna_1"
    frequency_mhz:  float = 900.0       # MHz
    tx_power_dbm:   float = 40.0        # dBm
    height_m:       float = 30.0        # meters above ground
    gain_dbi:       float = 0.0         # dBi
    tilt_deg:       float = 0.0         # electrical/mechanical tilt in degrees
    azimuth_deg:    float = 0.0         # pointing azimuth direction in degrees
    downtilt_deg:   float = 0.0         # mechanical/electrical downtilt in degrees
    beamwidth_h:    float = 360.0       # horizontal beamwidth (degrees); 360 = omni
    beamwidth_v:    float = 8.0         # vertical beamwidth (degrees)
    efficiency:     float = 0.6         # radiation efficiency, 0-1
    lat:            Optional[float] = None   # antenna latitude
    lng:            Optional[float] = None   # antenna longitude
    alt:            Optional[float] = None   # antenna altitude
    pos:            Optional[tuple] = None
    polarization:   str   = "vertical"    # 'vertical'|'horizontal'|'dual'
    pattern:        str   = "omni"        # antenna pattern identifier or filepath
    pattern_file:   Optional[str] = None

    def to_dict(self):
        return asdict(self)


# ─── Dispatcher ────────────────────────────────────────────────────────────────

def parse_antenna_config(source):
    """
    Main entry point. Accepts:
      - str: path to a file (.csv, .json, .xml, .kml) or other supported format
      - dict: raw key-value configuration
      - AntennaConfig: passed through unchanged

    Returns: AntennaConfig
    """
    if isinstance(source, AntennaConfig):
        return source

    if isinstance(source, dict):
        return _from_dict(source)

    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"Antenna config file not found: {source}")
        ext = os.path.splitext(source)[1].lower()
        parsers = {
            ".json": _from_json,
            ".csv":  _from_csv,
            ".xml":  _from_xml,
            ".kml":  _from_kml,
        }
        parser = parsers.get(ext, _from_generic_text)
        return parser(source)

    raise TypeError(f"Unsupported antenna source type: {type(source)}")


# ─── Format-specific parsers ────────────────────────────────────────────────────

def _validate_antenna_config(cfg: AntennaConfig) -> AntennaConfig:
    """Validate a normalized antenna config and apply safe defaults."""
    if cfg.frequency_mhz <= 0:
        raise ValueError("frequency_mhz must be positive")
    if cfg.tx_power_dbm < 0:
        raise ValueError("tx_power_dbm must be non-negative")
    if not (0 <= cfg.gain_dbi <= 50):
        raise ValueError("gain_dbi must be between 0 and 50")
    if cfg.height_m < 0:
        raise ValueError("height_m must be non-negative")
    if cfg.beamwidth_h <= 0:
        cfg.beamwidth_h = 360.0
    if cfg.beamwidth_v <= 0:
        cfg.beamwidth_v = 8.0
    return cfg


def _from_dict(d: dict) -> AntennaConfig:
    """Build an AntennaConfig from a plain dictionary, with safe key mapping."""
    # Support alternate key names from different tools/exports
    aliases = {
        "freq": "frequency_mhz",
        "frequency": "frequency_mhz",
        "power": "tx_power_dbm",
        "tx_power": "tx_power_dbm",
        "height": "height_m",
        "gain": "gain_dbi",
        "tilt": "tilt_deg",
        "beamwidth": "beamwidth_h",
        "latitude": "lat",
        "longitude": "lng",
    }
    normalized = {}
    for k, v in d.items():
        key = aliases.get(k.lower().strip(), k.lower().strip())
        normalized[key] = v

    cfg = AntennaConfig()
    for f, field_info in cfg.__dataclass_fields__.items():
        if f in normalized:
            try:
                field_type = field_info.type
                value = normalized[f]
                origin = getattr(field_type, '__origin__', None)
                if field_type in (float, int, bool):
                    if field_type is float:
                        setattr(cfg, f, float(value))
                    elif field_type is int:
                        setattr(cfg, f, int(value))
                    else:
                        setattr(cfg, f, str(value).lower() in ("1", "true", "yes"))
                elif origin is Optional:
                    inner = field_type.__args__[0]
                    if inner is float:
                        setattr(cfg, f, float(value))
                    elif inner is int:
                        setattr(cfg, f, int(value))
                    elif inner is bool:
                        setattr(cfg, f, str(value).lower() in ("1", "true", "yes"))
                    else:
                        setattr(cfg, f, value)
                else:
                    setattr(cfg, f, value)
            except (ValueError, TypeError) as e:
                print(f"WARNING: Could not set field '{f}': {e}")
    return _validate_antenna_config(cfg)


def _from_json(path: str) -> AntennaConfig:
    with open(path, "r") as f:
        data = json.load(f)
    # Support single object or list of objects (first used)
    if isinstance(data, list):
        data = data[0]
    print(f"Loaded antenna config from JSON: {path}")
    return _from_dict(data)


def _from_csv(path: str) -> AntennaConfig:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV file is empty: {path}")
    # Use first row
    print(f"Loaded antenna config from CSV: {path}")
    return _from_dict(dict(rows[0]))


def _from_xml(path: str) -> AntennaConfig:
    tree = ET.parse(path)
    root = tree.getroot()
    data = {}
    for child in root:
        data[child.tag] = child.text
    print(f"Loaded antenna config from XML: {path}")
    return _from_dict(data)


def _from_kml(path: str) -> AntennaConfig:
    """
    Parse a KML file for a Point placemark and extract extended data.
    KML namespace: http://www.opengis.net/kml/2.2
    """
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    tree = ET.parse(path)
    root = tree.getroot()
    data = {}

    # Extract coordinates
    coords_el = root.find(".//kml:coordinates", ns)
    if coords_el is not None and coords_el.text:
        parts = coords_el.text.strip().split(",")
        if len(parts) >= 2:
            data["lng"] = parts[0].strip()
            data["lat"] = parts[1].strip()

    # Extract ExtendedData SimpleData fields
    for sd in root.findall(".//kml:SimpleData", ns):
        data[sd.get("name", "")] = sd.text

    print(f"Loaded antenna config from KML: {path}")
    return _from_dict(data)


def _from_generic_text(path: str) -> AntennaConfig:
    """
    Fallback: try to parse a key=value or key:value text file.
    """
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
    print(f"Loaded antenna config from text file: {path}")
    return _from_dict(data)
