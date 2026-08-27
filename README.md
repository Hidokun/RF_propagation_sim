# RF Simulator

An end-to-end radio-frequency propagation simulator: model path loss over real
terrain, real building footprints and weather, then explore the results on an
interactive map. Ships with a **CLI**, a **Python API**, and a **Streamlit web UI**.

```
CLI / API                          Streamlit UI
   |                                   |
   v                                   v
[config parsers] --> [coverage engine] --> [folium map + reports]
        |                  |
        v                  v
[7 propagation models, DEM terrain, OSM buildings, weather terms]
```

---

## Table of Contents

- [Features](#features)
- [Quickstart](#quickstart)
- [Running It](#running-it)
  - [CLI](#cli)
  - [Web UI](#web-ui)
  - [Python API](#python-api)
- [Propagation Models](#propagation-models)
- [The Coverage Engine](#the-coverage-engine)
  - [Link Budget](#link-budget)
  - [Zones & Thresholds](#zones--thresholds)
  - [Terrain Diffraction](#terrain-diffraction)
  - [Buildings](#buildings)
  - [Combining Modes](#combining-modes)
  - [Dual Engines: Scalar vs Vector](#dual-engines-scalar-vs-vector)
- [Receivers](#receivers)
- [Configuration Reference](#configuration-reference)
- [External Data & Services](#external-data--services)
- [Outputs](#outputs)
- [Antenna Placement Optimization](#antenna-placement-optimization)
- [Testing](#testing)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Known Limitations & Gotchas](#known-limitations--gotchas)
- [Security Notes](#security-notes)
- [Credits](#credits)

---

## Features

- **7 propagation models**: FSPL, Close-In (CI), Longley-Rice (ITM), rain, atmospheric gas, fog, and Sionna ray tracing (GPU-optional with automatic FSPL fallback).
- **Real terrain**: SRTM DEM tiles (a verified Casablanca tile is bundled); knife-edge diffraction with a 4/3-earth bulge model anywhere, native Longley-Rice terrain profiles for links >= 1 km.
- **Real buildings**: OpenStreetMap footprints fetched via `osmnx`; wall-crossing and indoor losses.
- **Weather link budgets**: ITU-style rain/gas/fog attenuation applied outdoors-only on top of an FSPL base.
- **Bounded-box coverage grids** computed by two interchangeable engines (a readable scalar reference and a NumPy/shapely vectorized one, pinned equivalent to <= 0.01 dB).
- **Interactive map UI**: click to place transmitters/receivers, zone-painted coverage raster, hover RSSI popups, GeoJSON export.
- **Receiver reports**: point-by-point RSSI with serving-antenna attribution from CSV/KML/JSON inputs or map clicks.
- **Antenna placement optimizer** (SciPy L-BFGS-B) maximizing coverage uniformity.

---

## Quickstart

Requirements: **Python 3.13+** (developed on 3.13.5), `pip`.

```bash
# 1. Clone and create a virtual environment
git clone <your-fork-url> RF_Simulator
cd RF_Simulator
python -m venv .venv
.\.venv\Scripts\activate            # Windows (bash: source .venv/bin/activate)

# 2. Install backend deps (UI deps optional but recommended)
pip install -r RF_prop_sim/requirements.txt
pip install -r UI/requirements.txt

# For an exact, frozen reproducible environment:
pip install -r requirements-lock.txt

# 3. (Optional) configure API keys — see External Data & Services
copy .env.example .env               # then edit: GOOGLE_MAPS_API_KEY=...

# 4. Smoke test
python -m RF_prop_sim.main --address "Casablanca, Morocco" --model fspl --frequency_mhz 900 --distance_km 1 --report
streamlit run UI/app.py             # then open http://localhost:8501
```

No API keys are required to run: geocoding falls back to Casablanca,
buildings come from OpenStreetMap (no key), and the DEM tile ships in-repo.

---

## Running It

### CLI

```bash
# All flags
python -m RF_prop_sim.main --help

# From a config file (examples/ has ready-made ones)
python -m RF_prop_sim.main --config examples/sample_sim_config.json --report

# Ad-hoc single-path simulation
python -m RF_prop_sim.main --address "San Francisco, CA" --model itm \
    --frequency_mhz 900 --distance_km 3 --tx_height_m 30 --rx_height_m 1.5 --report

# Weather models (bare weather attenuation semantics on the CLI)
python -m RF_prop_sim.main --model rain --rain_rate_mmh 25 --frequency_mhz 18000 --distance_km 5

# Antenna placement optimization
python -m RF_prop_sim.main --optimize --opt_area_km 2 --frequency_mhz 3500
```

Key flags: `--config`, `--antenna`, `--address`, `--model {fspl,rain,ci,itm,sionna,gas,fog}`,
`--frequency_mhz`, `--distance_km`, `--tx_height_m/--rx_height_m`,
`--rain_rate_mmh [--rain_k --rain_alpha]`, `--antenna_power_dbm`, `--antenna_gain_dbi`,
`--antenna_lat/--antenna_lng/--antenna_alt_m`, `--optimize`, `--show_dem`, `--report`.

`--report` writes `report.md` (plus a Google Maps viewer when a key is present)
to `RF_prop_sim/Output/<UTC-timestamp>/`.

> **Semantics note**: the CLI reports the *bare* weather term for rain/gas/fog
> (e.g. "0.12 dB"). The coverage engine instead builds a full link budget
> (FSPL base + terrain + buildings + weather outdoors-only). Both are correct;
> they answer different questions.

### Web UI

```bash
streamlit run UI/app.py
```

Workflow:

1. Set the map centre (or keep the default) and optionally press **Re-centre Map**.
2. Click **Add Antenna**, then click the map to place it; switch its *nature* to
   `receiver` to place receivers instead. Edit per-antenna frequency/power/gain/
   height in the sidebar (each antenna keeps stable editor widgets).
3. Choose model, combining mode, grid resolution (10-100 m) and analysis box
   size (250-10 000 m). Configure weather sliders if using rain/gas/fog.
4. Press **Run Simulation**. The map paints a zone raster (one `ImageOverlay`),
   keeps hover/click RSSI popups via a sparse marker lattice, and shows zone
   statistics plus per-receiver reports.
5. Export the run as GeoJSON (`rf_coverage.geojson`) — the file embeds the exact
   run parameters used.

### Python API

```python
from RF_prop_sim.main import run_simulation
from RF_prop_sim.input_data_collection.ingestion import parse_simulation_config
from RF_prop_sim.antenna_data.parser import parse_antenna_config

cfg = parse_simulation_config({
    "address": "Casablanca, Morocco",
    "model": "itm",
    "frequency_mhz": 900.0,
    "distance_km": 2.0,
})
ant = parse_antenna_config("examples/sample_antenna.json")
result = run_simulation(cfg, ant)
print(result["path_loss_db"], result["location"])
```

For area studies use the engine directly:

```python
from RF_prop_sim.coverage_engine import compute_coverage_result, evaluate_receivers

res = compute_coverage_result(
    antennas=[{"name": "TX1", "lat": 33.58831, "lng": -7.61138,
               "frequency_mhz": 900, "tx_power_dbm": 40, "gain_dbi": 12,
               "height_m": 30, "nature": "transmitter"}],
    center_lat=33.58831, center_lng=-7.61138,
    box_size_m=2000, resolution_m=25,
    model="fspl", combining="superposition",   # engine="vector" is the default
)
# res["rssi"]      -> (H, W) dBm matrix          res["zone_code"] -> int8 0/1/2
# res["lats/lngs"] -> axis arrays                res["bounds"]    -> (S,N,W,E)
# res["stats"]     -> zone statistics            res["points_md5"]-> content hash
```

> Always construct configs via `parse_simulation_config()` /
> `parse_antenna_config()`. Calling the `SimulationConfig` dataclass directly
> with a path silently stores the path string as the address.

---

## Propagation Models

Canonical implementations live in
`RF_prop_sim/propagation_model/empirical_models.py` (+ `itm_model.py`,
`ray_tracing_model.py`). **Mind the unit split**: geometry models take
**MHz**, weather/ray-tracing models take **GHz**.

| Model | Units | Formula | Notes |
|---|---|---|---|
| `fspl` | MHz, km | `32.44 + 20log10(d_km) + 20log10(f_MHz)` | Free-space baseline |
| `ci` | MHz, m | `FSPL(d0) + 10n*log10(d/d0)` | Anchored at d0 so CI(n=2) == FSPL exactly for any reference distance |
| `rain` | **GHz**, km, mm/h | `k*R^alpha * d` (ITU-R P.838 power law) | Returns exactly 0 dB at R=0; custom `k`/`alpha` honored (partial overrides allowed) |
| `gas` | **GHz**, km | oxygen + water-vapor specific attenuation | Humidity is a **percent** (0-100); Tetens saturation vapor pressure |
| `fog` | **GHz**, km, g/m^3 | `0.2*M*f^2/(f^2+0.7) * d` (ITU-R P.840 style) | M = liquid water density |
| `itm` | MHz, km | Longley-Rice via the `itmlogic` package | Real terrain profiles >= 1 km; see below |
| `sionna` | **GHz** | Ray tracing (Sionna RT, max depth 3) | Needs GPU + LLVM-C.dll; otherwise falls back to FSPL over true positions |

All empirical models guard invalid inputs (`frequency <= 0`, `distance <= 0`)
and return `0.0`; the ITM wrapper raises `ValueError` instead (it validates
heights too).

### ITM / Longley-Rice details

- In the PyPI `itmlogic` port, `aref` is **excess** attenuation over free
  space; final loss = `FSPL + max(aref, 0)`.
- Terrain profiles passed as `pfl` must be `[int_count, spacing_m, e0..en]` —
  meters throughout.
- **Validity floor**: Longley-Rice assumes paths >= ~1 km. Sub-kilometre ITM
  links are served `FSPL + profile knife-edge` instead (smooth and physical).
  This holds in both engines, with or without a DEM.
- If `itmlogic` is not installed, a validated fallback (FSPL + horizon
  diffraction + terrain-type penalty) runs and the reason is recorded.
- itmlogic's reliability flag (`kwx >= 3`) surfaces as a user-facing warning in
  coverage results rather than being swallowed.

---

## The Coverage Engine

`RF_prop_sim/coverage_engine.py` computes signal-strength zones over a
**bounded box**: a square you define by centre, side length and resolution.
Every transmitter contributes to every grid point — there are no per-TX radius
limits and no sentinel holes; distant areas simply decay into worse zones.
Grid generation is index-based (`n = round(size/res) + 1`), so endpoints are
always included without float drift.

### Link Budget

Per transmitter-to-point path (outdoor case):

```
RSSI = tx_power_dbm + gain_dbi
       - base_model_loss(frequency, distance[, heights])
       - weather_attenuation        (rain/gas/fog models only, outdoor only)
       - terrain_diffraction        (knife-edge; non-ITM models)
       - building_walls + indoor    (if applicable)
```

Weather models automatically include an FSPL geometry base inside the engine,
so their maps are realistic link budgets (this differs from the CLI's bare
weather number by design).

Default weather posture everywhere is **"no weather unless configured"**:
rain defaults to 0 mm/h (returns exactly 0 dB), gas defaults to standard
atmosphere (15 C, 1013.25 hPa, 50 % RH), fog to 0.05 g/m^3.

### Zones & Thresholds

| Zone | RSSI | Colour | Code |
|---|---|---|---|
| good | >= -80 dBm | green `#22c55e` | 2 |
| medium | -95 .. -80 dBm | orange `#f97316` | 1 |
| bad | < -95 dBm | red `#ef4444` | 0 |

### Terrain Diffraction

- `_TerrainSampler` reads the DEM raster once into a process-wide singleton;
  nodata holes are filled with the scene median and out-of-bounds lookups
  clamp to the raster edge (the strict, raising variant for app code is
  `mapping/dem_provider.py::DemProvider`).
- Single knife-edge diffraction (continuous ITU-Lee variant, capped at 30 dB):
  worst Fresnel obstruction along a 17-sample path, earth bulge added with a
  4/3 effective radius factor.
- Near-field guard: the first/last 10% of each path is excluded and endpoint
  distances floored at 100 m, avoiding Fresnel-divergence blowups.
- Wavelength identity used throughout: `lambda(m) = 299.792458 / f_MHz`.
- ITM links >= 1 km skip the knife-edge and feed the real profile to
  Longley-Rice.

### Buildings

OSM footprints are indexed in a shapely STRtree (`_BuildingIndex`). Losses:

- first wall crossed: **+12 dB**
- each additional wall: **+6 dB** (capped at **30 dB** total)
- receiver indoors (point-in-polygon): additional **+15 dB**

Vector bulk operations count walls for the whole grid in one STRtree query.
Coordinate order matters: shapely wants `(x=lng, y=lat)` — swapping them
silently zeroes every building effect.

### Combining Modes

With multiple transmitters, per-point signals combine as:

- `superposition` *(default)* — power sum: `RSSI = 10log10(sum(10^(RSSI_i/10)))`
- `best_server` — strongest server only (handover-style)

### Dual Engines: Scalar vs Vector

| Engine | Role | Implementation |
|---|---|---|
| `scalar` | readable per-link reference | plain Python loops |
| `vector` *(default)* | production fast path | NumPy matrices + shapely-2 bulk ops |

Equivalence is pinned by tests (`TestVectorEquivalence`): <= 0.01 dB element
agreement, identical zone codes and best-server attribution across models,
combining modes and +/- buildings.

---

## Receivers

Receivers are points (CSV rows, KML/JSON entries, or UI-placed markers of
nature `receiver`). `evaluate_receivers()` evaluates **every** receiver against
**every** transmitter and returns per-receiver:

```json
{"name": "RX1", "lat": ..., "lng": ..., "rssi_dbm": -71.3,
 "zone": "good", "color": "#22c55e",
 "serving_antenna": "TX1", "covered": true}
```

- `covered` means `rssi_dbm >= -95` (medium threshold).
- A per-receiver `height_m` is honored end-to-end (default 1.5 m).
- User-configured weather parameters and the combining mode are forwarded into
  these budgets.

Multi-row CSV files are interpreted as receiver grids and require `lat` +
(`lng`|`longitude`|`long`) columns; malformed rows are tallied and reported,
never silently dropped.

---

## Configuration Reference

`SimulationConfig` highlights (see `input_data_collection/ingestion.py` for the
full list): `address`, `center_lat/lng`, `area_radius_m` (building-fetch
radius), `model`, `frequency_mhz` (900), `distance_km` (0.5),
`tx_height_m` (30), `rx_height_m` (1.5), `rain_rate_mmh` (0.0), `rain_k`,
`rain_alpha`, `fog_liquid_water_density_gm3` (0.05), `temperature_c` (15),
`pressure_hpa` (1013.25), `relative_humidity` (50 %), ITM set
(`surface_refractivity`, `effective_earth_radius_factor`,
`ground_permittivity`, `ground_conductivity`, `terrain_type`), CI set
(`ci_path_loss_exponent`, `ci_reference_distance_m`), `tx_power_dbm` (40),
`antenna_gain_dbi`, `combining` (`superposition` | `best_server`),
`run_optimization`, `run_dem`, `output_dir`, `receivers[]`.

Accepted alias spellings include `freq/frequency -> frequency_mhz`,
`dist/distance -> distance_km`, `humidity -> relative_humidity`,
`fog/fog_density -> fog_liquid_water_density_gm3` (legacy keys remapped),
`optimize -> run_optimization`, and more.

Input formats: JSON object, CSV (single row = config; multi-row = receivers),
XML elements, KML (`<coordinates>` + `SimpleData`), and generic `key=value`
text lines. Validation rejects non-positive frequency/distance, negative
heights/rain, humidity outside 0-100, out-of-range coordinates; unknown models
fall back to `fspl` with a warning.

Antenna configs (`antenna_data/parser.py`) accept the same formats with fields
`name, frequency_mhz, tx_power_dbm, height_m, gain_dbi, tilt_deg, azimuth_deg,
beamwidth_h/v, efficiency, lat/lng/alt, polarization, pattern`. CSV antennas
use the first row only.

---

## External Data & Services

| Service | Key needed? | Behaviour without it |
|---|---|---|
| Google geocoding | yes (`GOOGLE_MAPS_API_KEY` in `.env`, or `google_key.json`) | Falls back to Casablanca (33.5883, -7.61138) |
| OpenStreetMap buildings | no | `osmnx` fetch fails gracefully -> None (no building losses) |
| DEM terrain | no | Bundled real SRTM tile `data/dem/casablanca_srtm.tif`; engine degrades to flat earth if absent |

Fetch a real DEM anywhere (AWS Terrarium tiles, no API key):

```bash
python scripts/fetch_srtm_dem.py --west -7.75 --south 33.5 --east -7.45 --north 33.7 \
    --zoom 12 --output data/dem/my_area.tif
```

Set `DEM_PATH=/custom/dir` in `.env` to relocate the DEM directory. Real tiles
are always preferred over synthetic fallbacks; if the code ever falls back to
the synthetic sample it prints a loud warning.

Optional heavy extras: `sionna-rt` additionally needs an NVIDIA GPU and
`LLVM-C.dll` at runtime (imports succeed without them; features degrade);
`pyvista` powers the optional `--show_dem` 3D preview window.

---

## Outputs

| Artifact | Where | Contents |
|---|---|---|
| `report.md` | `RF_prop_sim/Output/<timestamp>/` | UTC date, summary bullets (status/model/path loss/elevation/location/antenna) |
| `coverage_map.html` | output dir | Static folium map (CartoDB positron) with buildings, antennas, coverage circles |
| `map_view.html` + `maps.js` | same, only with a Google key present | Interactive viewer; **embeds your API key in plain text** — treat as sensitive |
| `rf_coverage.geojson` | UI download button | Coverage points + metadata snapshot of the exact run parameters |
| `scene_3d_render.png` | CWD (sionna CLI runs) | Ray-tracing scene render |

---

## Antenna Placement Optimization

```bash
python -m RF_prop_sim.main --optimize
```

`optimization/optimizer.py` searches `(x_km, y_km, height_m)` with SciPy
L-BFGS-B (height bounded 10-100 m) over a 10x10 receiver grid, minimizing

```
cost = sum(max(0, target - Prx)^2)          # under-coverage
     + 0.5 * sum(max(0, Prx - target-10dB)^2) # over-coverage past target+10 dB
     + 10 * Var(Prx)                        # uniformity
target = -80 dBm
```

It uses a pure-FSPL proxy (no terrain/buildings/weather) and is currently a
CLI-only feature.

---

## Testing

Run from the `RF_prop_sim/` directory (its `pytest.ini` is the active config;
the venv path below assumes the project-root `.venv`):

```bash
cd RF_prop_sim
..\.venv\Scripts\python.exe -m pytest tests/
..\.venv\Scripts\python.exe -m pytest tests/ -m propagation           # one marker
..\.venv\Scripts\python.exe -m pytest tests/ -m "not requires_itmlogic and not requires_sionna"
..\.venv\Scripts\python.exe -m pytest tests/propagation/test_coverage_physics.py -v
```

- Markers: `propagation`, `integration`, `edge_case`, `performance`, `slow`,
  `requires_itmlogic`, `requires_sionna`. Tests are auto-marked by directory.
- Current state: **209 passed / 3 skipped / 0 failed** (skips = GPU-bound
  Sionna cases).
- Physics is validated against closed-form expressions, never against another
  implementation. Keystone suites: `test_coverage_physics.py`
  (vector/scalar equivalence, bounded-box radial continuity, audit
  regressions), `test_ci_anchor.py` (CI==FSPL anchoring).
- Performance tests use best-of-N sampling and deliberately loose (5x)
  regression bounds — tight ms-ratio bounds flake on shared hardware.

Engine benchmark:

```bash
python scripts/bench_engines.py
```

---

## Performance

Measured on the dev laptop (2 km box @ 25 m => 81x81 = 6561 links):

| Scenario | Scalar | Vector | Speedup |
|---|---|---|---|
| FSPL, no buildings | 615 ms | 26 ms | ~24x |
| FSPL + buildings | 1857 ms | 123 ms | ~15x |
| 5 km box @ 50 m, FSPL + buildings | 2994 ms | 211 ms | ~14x |

Map rendering scales independently: one RGBA `ImageOverlay` replaces thousands
of circle markers (hover popups survive via a <= ~600-dot lattice).

---

## Project Structure

```
RF_Simulator/
├── RF_prop_sim/                 # backend package (import via sys.path, not pip-installed)
│   ├── main.py                  # CLI + run_simulation() API entry points
│   ├── coverage_engine.py       # bounded-box coverage: scalar/vector engines, terrain, buildings
│   ├── propagation_model/       # empirical_models.py, itm_model.py, ray_tracing_model.py
│   ├── input_data_collection/   # ingestion.py: SimulationConfig parser (JSON/CSV/XML/KML/text)
│   ├── antenna_data/            # antenna config parser
│   ├── mapping/                 # geocoding, osmnx buildings, DEM provider, folium/pyvista viz
│   ├── optimization/            # L-BFGS-B placement optimizer
│   ├── Output/<timestamp>/      # generated reports and maps
│   └── tests/                   # pytest suite (auto-marked by directory)
├── UI/app.py                    # Streamlit application (single file)
├── examples/                    # ready-made simulation + antenna configs
├── scripts/
│   ├── fetch_srtm_dem.py        # AWS Terrarium DEM downloader (no key)
│   └── bench_engines.py         # scalar/vector micro-benchmark
├── data/dem/                    # DEM rasters (real Casablanca SRTM tile included)
├── config.py                    # API keys (.env), DEM paths, fallbacks
└── requirements-lock.txt        # exact freeze incl. transitive deps
```

---

## Known Limitations & Gotchas

1. **Unit split by design**: geometry models take MHz, weather and ray tracing
   take GHz. Config fields are `*_mhz`; parsers normalize aliases.
2. **CLI vs engine weather semantics**: CLI prints bare weather attenuation;
   the coverage engine builds full link budgets (FSPL base + terrain +
   buildings + weather outdoors-only). Indoor receivers intentionally receive
   no weather term.
3. **ITM sub-km floor**: Longley-Rice is only used for paths >= 1 km; shorter
   ITM links get FSPL + knife-edge. Exact-1 km routing parity is tested.
4. **Engine `"sionna"` is an analytic surrogate** (FSPL + urban excess), not
   ray tracing; real RT runs only through the CLI single-link path. The UI
   model list omits sionna for this reason.
5. **Shapely coordinate order** is `(lng, lat)`; the engines handle this, but
   keep it in mind when touching geometry code.
6. **st_folium mutates folium maps in place** — the UI passes a
   `copy.deepcopy()` of its cached map and remounts the widget on content
   changes; don't bypass that machinery casually.
7. The optimizer ignores terrain/buildings/weather (FSPL proxy) and silently
   returns its initial guess if convergence fails.
8. Text-format configs split on the first `=` or `:` — addresses containing
   `:` will corrupt such files.
9. No `.env` in the repo: create your own from `.env.example`. A malformed
   `google_key.json` is tolerated but silently disables geocoding (fallback
   coordinates are used).

---

## Security Notes

- Put secrets in `.env` (gitignored). `google_key.json` is also gitignored but
  currently optional legacy support.
- Generated `map_view.html` artifacts embed the Google Maps key **in plain
  text** so the HTML is shareable standalone. Only share those files with
  audiences that may hold the key, and restrict the key by HTTP referrer in
  the Google Cloud console.
- Reports/maps are written under `RF_prop_sim/Output/`; review before
  publishing any generated folder.

---

## Credits

- Terrain: AWS Open Data Terrain Tiles ("Terrarium") via `scripts/fetch_srtm_dem.py`;
  bundled Casablanca tile cross-checked against source.
- Buildings: OpenStreetMap contributors, via `osmnx`.
- Longley-Rice: the `itmlogic` Python port.
- Ray tracing: NVIDIA Sionna RT (scenes `munich`, `simple_street_canyon`).
- Basemaps: CARTO / OpenStreetMap via folium; sample DEM asset from rasterio.
- Atmospheric coefficients: simplified ITU-R P.838 / P.840-style models.

---

## References & Citations

1. Amazon Web Services (AWS), *Open Data Terrain Tiles (Terrarium)*. [Online]. Available: https://registry.opendata.aws/terrain-tiles/
2. OpenStreetMap contributors, *OpenStreetMap Data*. [Online]. Available: https://www.openstreetmap.org/
3. Boeing, G., *OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks*, Computers, Environment and Urban Systems, 65, 126-139, 2017.
4. Hufford, G. A., Longley, A. G., & Kissick, W. A., *A Guide to the Use of the ITS Irregular Terrain Model in the Area Prediction Mode*, NTIA Report 82-100, 1982. (Implemented via `itmlogic` Python port).
5. Hoydis, J., Cammerer, S., Ait Aoudia, F., Vem, A., Binder, N., Marcus, G., & Keller, A., *Sionna: An Open-Source Library for Next-Generation Physical Layer Research*. [Online]. Available: https://nvlabs.github.io/sionna/
6. ITU-R P.838-3, *Specific attenuation model for rain for use in prediction methods*.
7. ITU-R P.840-8, *Attenuation due to clouds and fog*.

