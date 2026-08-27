"""
UI/app.py — RF Propagation Simulator  (Streamlit + streamlit-folium)

Features:
  • Interactive Folium map embedded via streamlit-folium (view persists
    across reruns via a content-fingerprint cache)
  • "Add Antenna" button — click on the map to place an antenna instantly
  • Per-antenna sidebar config panel (power, gain, height, frequency, nature)
  • User-defined square Analysis Area: every transmitter contributes to every
    point inside the box (no per-transmitter radius limits)
  • Physics-aware link budgets: terrain diffraction + building walls +
    weather (outdoors only); superposition or best-server combining
  • Green / Orange / Red signal zones rendered on the map
  • Receiver Report + zone statistics + GeoJSON export
"""

import sys
import os
import json
import copy
import hashlib
import math
from uuid import uuid4

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RF_SIM = os.path.join(ROOT, "RF_prop_sim")
sys.path.insert(0, ROOT)
sys.path.insert(0, RF_SIM)

import streamlit as st

# ── Page config must be first Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="RF Propagation Simulator",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Deferred heavy imports (after path setup) ─────────────────────────────────
import folium
from streamlit_folium import st_folium

from RF_prop_sim.coverage_engine import (
    compute_coverage_grid,
    compute_coverage_result,
    evaluate_receivers,
    zone_statistics,
    RSSI_GOOD,
    RSSI_MEDIUM,
    ZONE_COLORS,
    box_bounds,
)
from RF_prop_sim.mapping.terrain_builder import download_buildings

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.2);
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px;
    padding: 14px !important;
    backdrop-filter: blur(8px);
}
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #a5b4fc !important;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.6) !important;
}

/* Secondary button */
.stButton > button:not([kind="primary"]) {
    background: rgba(99,102,241,0.15) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    border-radius: 8px !important;
    color: #a5b4fc !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:not([kind="primary"]):hover {
    background: rgba(99,102,241,0.3) !important;
    transform: translateY(-1px) !important;
}

/* Info/success/warning/error boxes */
.stAlert {
    border-radius: 10px !important;
}

/* Headers */
h1 { color: #a5b4fc !important; font-weight: 700 !important; }
h2 { color: #c7d2fe !important; font-weight: 600 !important; }
h3 { color: #e2e8f0 !important; font-weight: 600 !important; }

/* Divider */
hr { border-color: rgba(99,102,241,0.2) !important; }

/* Expander */
.streamlit-expanderHeader {
    background: rgba(99,102,241,0.12) !important;
    border-radius: 8px !important;
}

/* Number inputs / selects */
[data-testid="stNumberInput"], [data-testid="stSelectbox"] {
    background: rgba(15,23,42,0.5) !important;
}

/* Zone stat cards */
.zone-card {
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    font-family: 'Inter', sans-serif;
}
.zone-good  { background: rgba(34,197,94,0.15);  border: 1px solid rgba(34,197,94,0.4);  }
.zone-med   { background: rgba(249,115,22,0.15); border: 1px solid rgba(249,115,22,0.4); }
.zone-bad   { background: rgba(239,68,68,0.15);  border: 1px solid rgba(239,68,68,0.4);  }

/* Antenna list entry */
.ant-card {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# ── Session-state initialisation ──────────────────────────────────────────────
def _init():
    defaults = {
        "antennas": [],           # list of antenna dicts
        "placing_mode": False,    # True while waiting for a map click
        "coverage_points": [],    # output of compute_coverage_grid
        "coverage_result": None,  # matrices + bounds + points_md5 (raster path)
        "zone_stats": {},         # output of zone_statistics
        "map_center": [33.5883, -7.61138],
        "map_zoom": 15,
        "sim_ran": False,
        "buildings_gdf": None,
        "buildings_meta": None,   # {"lat","lng","dist"} of the fetch center
        # Default params for new antennas
        "def_freq": 900.0,
        "def_power": 40.0,
        "def_gain": 12.0,
        "def_height": 30.0,
        "box_size_m": 2000.0,          # user-defined square analysis area side
        "def_model": "fspl",
        "def_nature": "transmitter",   # New: default nature for new antennas
        "grid_res_m": 25.0,            # coverage grid resolution (m)
        "combining": "superposition",  # overlap physics mode
        "rain_rate_mmh": 0.0,          # no weather unless configured (matches CLI default)
        "relative_humidity": 50.0,
        "fog_liquid_water_density_gm3": 0.05,
        "temperature_c": 15.0,
        "receiver_reports": [],        # per-receiver RSSI results
        # Map persistence: cache the built folium object keyed by a content
        # fingerprint so st_folium never remounts on passive reruns (which
        # would snap the Leaflet view back to the stored center).
        "folium_cache": None,          # {"fingerprint", "map"}
        "view_epoch": 0,               # bumped only by explicit recenter/placement
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init()


# ── Helper ────────────────────────────────────────────────────────────────────
ANT_COLORS = ["red", "blue", "purple", "orange", "darkred",
              "lightred", "darkblue", "darkgreen", "cadetblue", "pink"]


def _build_folium_map(center, zoom, antennas, coverage_points,
                      buildings_gdf=None, receiver_reports=None, box_size_m=None,
                      coverage_result=None):
    """Build and return a folium.Map with current state."""
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")

    # Analysis-area preview: dashed outline of the user-defined square so the
    # computation region is visible before running.
    if box_size_m and antennas:
        try:
            b_min_lat, b_min_lng, b_max_lat, b_max_lng = box_bounds(
                center[0], center[1], float(box_size_m))
            folium.Rectangle(
                bounds=[[b_min_lat, b_min_lng], [b_max_lat, b_max_lng]],
                color="#94a3b8", weight=1.5, dash_array="6,6",
                fill=True, fill_color="#94a3b8", fill_opacity=0.02,
                tooltip=f"Analysis Area ({float(box_size_m):.0f} m side)",
            ).add_to(m)
        except Exception:
            pass

    # OSM building footprints (context + the geometry used for wall losses)
    if buildings_gdf is not None and len(buildings_gdf) > 0:
        try:
            gj = folium.GeoJson(
                buildings_gdf.to_json(),
                name="Buildings",
                style_function=lambda _f: {
                    "fillColor": "#475569", "color": "#64748b",
                    "weight": 0.5, "fillOpacity": 0.35,
                },
                tooltip=folium.GeoJsonTooltip(fields=[], labels=False),
            )
            gj.add_to(m)
        except Exception:
            pass

    # Coverage zones — ONE raster overlay (constant cost at any grid size)
    # plus a bounded, invisible interaction lattice that preserves per-point
    # hover/click RSSI popups.
    if coverage_result is not None and coverage_result.get("rssi") is not None:
        import numpy as np
        rssi = coverage_result["rssi"]
        zone_code = coverage_result["zone_code"]
        b_min_lat, b_min_lng, b_max_lat, b_max_lng = coverage_result["bounds"]
        H, W = rssi.shape

        cov_layer = folium.FeatureGroup(name="Coverage Zones")

        # 1) Raster: uint8 RGBA (branca passes 4-channel arrays through
        #    untouched; non-uint8 would get rescaled). Row 0 must be the
        #    northern edge -> flip the lat-ascending matrix vertically.
        palette = np.array([
            [239, 68, 68],     # zone 0 bad   #ef4444
            [249, 115, 22],    # zone 1 medium#f97316
            [34, 197, 94],     # zone 2 good  #22c55e
        ], dtype=np.uint8)
        rgba = np.empty((H, W, 4), dtype=np.uint8)
        rgba[:, :, :3] = palette[zone_code]
        rgba[:, :, 3] = 150                       # ~0.59 opacity, matches old fill
        rgba = np.flipud(rgba)

        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[[b_min_lat, b_min_lng], [b_max_lat, b_max_lng]],
            origin="upper",
            mercator_project=True,    # audit L-6: warp rows so large boxes align
            name="Coverage Zones",
            opacity=1.0,
        ).add_to(cov_layer)

        # 2) Interaction lattice: transparent wide-hit circles on a stride so
        #    marker count stays <=~600 regardless of resolution, keeping
        #    hover/click RSSI popups without O(points) render cost.
        stride = max(1, int(math.ceil((max(H, W) - 1) / 23.0)))
        pts = coverage_result["points"]          # row-major lat asc, lng asc
        idx = 0
        for i in range(0, H, stride):
            for j in range(0, W, stride):
                pt = pts[i * W + j]
                folium.CircleMarker(
                    location=[pt["lat"], pt["lng"]],
                    radius=9,
                    color="#000000",
                    opacity=0.0,
                    weight=8,
                    fill=False,
                    fill_opacity=0.0,
                    popup=folium.Popup(
                        f"<b>RSSI:</b> {pt['rssi_dbm']} dBm<br>"
                        f"<b>Zone:</b> {pt['zone'].capitalize()}<br>"
                        f"<b>Best:</b> {pt.get('best_antenna', '—')}",
                        max_width=180,
                    ),
                    tooltip=f"{pt['rssi_dbm']} dBm · {pt['zone']}",
                ).add_to(cov_layer)
                idx += 1

        cov_layer.add_to(m)
    elif coverage_points:
        # Fallback path (no matrices available): legacy per-point circles.
        cov_layer = folium.FeatureGroup(name="Coverage Zones")
        for pt in coverage_points:
            folium.CircleMarker(
                location=[pt["lat"], pt["lng"]],
                radius=5,
                color=pt["color"],
                fill=True,
                fill_color=pt["color"],
                fill_opacity=0.55,
                weight=0,
                popup=folium.Popup(
                    f"<b>RSSI:</b> {pt['rssi_dbm']} dBm<br>"
                    f"<b>Zone:</b> {pt['zone'].capitalize()}<br>"
                    f"<b>Best:</b> {pt.get('best_antenna', '—')}",
                    max_width=180,
                ),
                tooltip=f"{pt['rssi_dbm']} dBm · {pt['zone']}",
            ).add_to(cov_layer)
        cov_layer.add_to(m)

    # Antenna markers
    if antennas:
        ant_layer = folium.FeatureGroup(name="Antennas")
        for i, ant in enumerate(antennas):
            color = ANT_COLORS[i % len(ANT_COLORS)]
# Different icon for transmitter vs receiver
            # Folium icon names: we'll use 'broadcast-tower' for transmitter and 'wifi' for receiver? 
            # But note: Folium's icon prefix 'fa' (Font Awesome) has:
            #   transmitter: fa-broadcast-tower (https://fontawesome.com/icons/broadcast-tower?s=solid)
            #   receiver: fa-wifi (https://fontawesome.com/icons/wifi?s=solid)
            # However, we don't have the exact names, but we can use:
            #   For transmitter: 'broadcast-tower' (if available) or 'signal'
            #   For receiver: 'wifi' or 'mobile-alt'
            # Let's use:
            #   transmitter: 'broadcast-tower' (if not available, fallback to 'signal')
            #   receiver: 'wifi'
            # But note: we are using the 'fa' prefix.
            # We'll check: 
            #   For transmitter: icon = folium.Icon(color=color, icon="broadcast-tower", prefix="fa")
            #   For receiver: icon = folium.Icon(color=color, icon="wifi", prefix="fa")
            # However, if the icon name is not found, it will default to a standard icon.
            # We'll use:
            icon = "broadcast-tower" if ant["nature"] == "transmitter" else "wifi"
            folium.Marker(
                [ant["lat"], ant["lng"]],
                icon=folium.Icon(color=color, icon=icon, prefix="fa"),
                tooltip=ant["name"],
                popup=folium.Popup(
                    f"<b>{ant['name']}</b><br>"
                    f" {ant['frequency_mhz']} MHz | {ant['tx_power_dbm']} dBm<br>"
                    f" Gain: {ant['gain_dbi']} dBi | H: {ant['height_m']} m<br>"
                    f" Nature: {ant['nature'].capitalize()}",
                    max_width=200,
                ),
            ).add_to(ant_layer)
        ant_layer.add_to(m)

    # Receiver markers colored by received signal quality
    if receiver_reports:
        rx_layer = folium.FeatureGroup(name="Receivers")
        for rx in receiver_reports:
            folium.CircleMarker(
                location=[rx["lat"], rx["lng"]],
                radius=8,
                color=rx["color"],
                fill=True,
                fill_color=rx["color"],
                fill_opacity=0.9,
                weight=2,
                popup=folium.Popup(
                    f"<b>{rx['name']}</b><br>"
                    f"<b>RSSI:</b> {rx['rssi_dbm']} dBm<br>"
                    f"<b>Zone:</b> {rx['zone'].capitalize()}<br>"
                    f"<b>Serving:</b> {rx['serving_antenna']}",
                    max_width=200,
                ),
                tooltip=f"{rx['name']} · {rx['rssi_dbm']} dBm · {rx['zone']}",
            ).add_to(rx_layer)
        rx_layer.add_to(m)

    # Legend (shown only after simulation)
    if coverage_points:
        legend = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:rgba(15,23,42,0.92);border-radius:10px;
                    padding:12px 18px;box-shadow:0 4px 20px rgba(0,0,0,0.5);
                    font-family:Inter,sans-serif;color:#f1f5f9;
                    border:1px solid rgba(255,255,255,0.1);">
          <p style="margin:0 0 8px 0;font-weight:700;font-size:13px;color:#94a3b8;">
               Signal Coverage</p>
          <div style="display:flex;align-items:center;margin-bottom:5px;">
            <span style="width:12px;height:12px;border-radius:50%;background:#22c55e;
                         display:inline-block;margin-right:8px;"></span>
            <span style="font-size:12px;">Good  (≥ −80 dBm)</span></div>
          <div style="display:flex;align-items:center;margin-bottom:5px;">
            <span style="width:12px;height:12px;border-radius:50%;background:#f97316;
                         display:inline-block;margin-right:8px;"></span>
            <span style="font-size:12px;">Medium (−80 → −95 dBm)</span></div>
          <div style="display:flex;align-items:center;">
            <span style="width:12px;height:12px;border-radius:50%;background:#ef4444;
                         display:inline-block;margin-right:8px;"></span>
            <span style="font-size:12px;">Bad  (< −95 dBm)</span></div>
        </div>"""
        m.get_root().html.add_child(folium.Element(legend))

    # Placement-mode hint marker
    if st.session_state.placing_mode:
        hint = """
        <div style="position:fixed;top:80px;left:50%;transform:translateX(-50%);
                    z-index:9999;background:rgba(99,102,241,0.95);border-radius:10px;
                    padding:10px 20px;color:white;font-family:Inter,sans-serif;
                    font-weight:600;box-shadow:0 4px 16px rgba(0,0,0,0.4);
                    pointer-events:none;">
            📍 Click anywhere on the map to place the antenna
        </div>"""
        m.get_root().html.add_child(folium.Element(hint))

    folium.LayerControl().add_to(m)
    return m


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("##  RF Propagation Simulator")
    st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)

    # ── Map centre ────────────────────────────────────────────────────────────
    st.markdown("### 🗺 Map Centre")
    c1, c2 = st.columns(2)
    with c1:
        ctr_lat = st.number_input("Latitude", value=st.session_state.map_center[0],
                                  format="%.5f", key="inp_lat")
    with c2:
        ctr_lng = st.number_input("Longitude", value=st.session_state.map_center[1],
                                  format="%.5f", key="inp_lng")

    if st.button(" Re-centre Map"):
        st.session_state.map_center = [ctr_lat, ctr_lng]
        # Center/zoom are excluded from the map fingerprint (so passive reruns
        # never rebuild); an epoch bump forces the one deliberate rebuild.
        st.session_state.view_epoch += 1
        st.rerun()

    st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)

    # ── Default antenna parameters ────────────────────────────────────────────
    st.markdown("###  Default Antenna Parameters")
    st.caption("Applied to each newly placed antenna")

    st.session_state.def_freq   = st.number_input("Frequency (MHz)", min_value=1.0,
                                                   value=st.session_state.def_freq, step=10.0)
    st.session_state.def_power  = st.number_input("TX Power (dBm)",
                                                   value=st.session_state.def_power, step=1.0)
    st.session_state.def_gain   = st.number_input("Gain (dBi)",
                                                   value=st.session_state.def_gain, step=0.5)
    st.session_state.def_height = st.number_input("Height (m)", min_value=1.0,
                                                   value=st.session_state.def_height, step=1.0)
    st.session_state.def_model  = st.selectbox(
        "Propagation Model",
        ["fspl", "ci", "itm", "rain", "gas", "fog"],
        index=["fspl", "ci", "itm", "rain", "gas", "fog"].index(st.session_state.def_model),
    )
    st.session_state.def_nature = st.selectbox(
        "Antenna Nature",
        ["transmitter", "receiver"],
        index=["transmitter", "receiver"].index(st.session_state.def_nature),
    )

    st.markdown("#### Grid Resolution")
    st.caption("Distance between coverage sample points")
    st.session_state.grid_res_m = st.slider(
        "Resolution (m)", min_value=10.0, max_value=100.0,
        value=float(st.session_state.grid_res_m), step=5.0,
        help="Finer grids give smoother maps but run slower.",
    )

    st.markdown("#### Overlap Physics")
    st.session_state.combining = st.radio(
        "Combining mode",
        ["superposition", "best_server"],
        index=["superposition", "best_server"].index(st.session_state.combining),
        help=("Superposition: overlapping signals add in the power domain "
              "(real receiver physics). Best-server: strongest transmitter wins."),
    )

    st.markdown("#### 🌦 Weather (outdoor paths)")
    st.caption("Applied when a weather model is selected; skipped for indoor points")
    st.session_state.rain_rate_mmh = st.number_input(
        "Rain rate (mm/h)", min_value=0.0, max_value=150.0,
        value=float(st.session_state.rain_rate_mmh), step=1.0)
    st.session_state.relative_humidity = st.number_input(
        "Relative humidity (%)", min_value=0.0, max_value=100.0,
        value=float(st.session_state.relative_humidity), step=5.0)
    st.session_state.fog_liquid_water_density_gm3 = st.number_input(
        "Fog density (g/m³)", min_value=0.0, max_value=2.0,
        value=float(st.session_state.fog_liquid_water_density_gm3), step=0.01,
        format="%.3f")
    st.session_state.temperature_c = st.number_input(
        "Temperature (°C)", min_value=-40.0, max_value=55.0,
        value=float(st.session_state.temperature_c), step=1.0)

    st.markdown("####  Analysis Area")
    st.caption("Square region centered on the map centre — every transmitter "
               "contributes to every point inside it")
    st.session_state.box_size_m = st.slider(
        "Box side length (m)", min_value=250.0, max_value=10000.0,
        value=float(st.session_state.box_size_m), step=250.0,
        help="Larger boxes and finer resolutions mean more computation. "
             "ITM over big boxes is slow; prefer FSPL/CI there.")

    st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)

# ── Antenna list ──────────────────────────────────────────────────────────
    st.markdown(f"### Placed Antennas ({len(st.session_state.antennas)})")
     
    if not st.session_state.antennas:
        st.info("No antennas yet. Click **Add Antenna** on the map panel.")
    else:
        for idx, ant in enumerate(st.session_state.antennas):
            # Stable per-antenna identity: widget keys must follow the
            # ANTENNA, not its list position. Positional keys made deletion
            # of antenna #0 silently overwrite every survivor's parameters
            # with the removed one's stale widget state.
            ant.setdefault("id", uuid4().hex[:8])
            uid = ant["id"]
            with st.expander(f"🔵 {ant['name']}", expanded=False):
                ant["frequency_mhz"] = st.number_input(
                    "Freq (MHz)", value=ant["frequency_mhz"],
                    key=f"freq_{uid}", step=10.0)
                ant["tx_power_dbm"]  = st.number_input(
                    "Power (dBm)", value=ant["tx_power_dbm"],
                    key=f"pwr_{uid}", step=1.0)
                ant["gain_dbi"]      = st.number_input(
                    "Gain (dBi)", value=ant["gain_dbi"],
                    key=f"gain_{uid}", step=0.5)
                ant["height_m"]      = st.number_input(
                    "Height (m)", value=ant["height_m"],
                    key=f"ht_{uid}", min_value=1.0, step=1.0)
                ant["nature"]        = st.selectbox(
                    "Nature", ["transmitter", "receiver"],
                    index=["transmitter", "receiver"].index(ant["nature"]),
                    key=f"nature_{uid}")
                st.caption(f"📍 {ant['lat']:.5f}, {ant['lng']:.5f}")
                if st.button("🗑 Remove", key=f"rm_{uid}"):
                    st.session_state.antennas.pop(idx)
                    st.session_state.coverage_points = []
                    st.session_state.coverage_result = None
                    st.session_state.sim_ran = False
                    st.rerun()

    st.markdown("<hr style='margin:8px 0'>", unsafe_allow_html=True)

    if st.button("🗑️ Clear All Antennas"):
        st.session_state.antennas = []
        st.session_state.coverage_points = []
        st.session_state.coverage_result = None
        st.session_state.zone_stats = {}
        st.session_state.sim_ran = False
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("#  RF Coverage Simulator")
st.markdown("Place antennas on the map, run the simulation, and see coverage zones in real time.")
st.markdown("---")

# ── Top action bar ─────────────────────────────────────────────────────────────
col_add, col_run, col_clear, col_spacer = st.columns([1, 1, 1, 3])

with col_add:
    if not st.session_state.placing_mode:
        if st.button(" Add Antenna", type="primary", use_container_width=True):
            st.session_state.placing_mode = True
            st.rerun()
    else:
        if st.button(" Cancel Placement", use_container_width=True):
            st.session_state.placing_mode = False
            st.rerun()

with col_run:
    run_disabled = not any(a["nature"] == "transmitter"
                           for a in st.session_state.antennas)   # audit L-8
    if st.button("▶ Run Simulation", type="primary",
                 disabled=run_disabled, use_container_width=True):
        with st.spinner(f"Computing coverage grid over a {st.session_state.box_size_m:.0f} m box "
                        f"({st.session_state.grid_res_m:.0f} m resolution)…"):
            # Filter antennas by nature
            transmitters = [ant for ant in st.session_state.antennas if ant["nature"] == "transmitter"]
            receivers = [ant for ant in st.session_state.antennas if ant["nature"] == "receiver"]

            # OSM building footprints: wall-penetration geometry + map polygons.
            # Re-fetch when the analysis center has moved too far from where
            # the current footprints were downloaded for — otherwise wall
            # losses would describe the WRONG neighborhood (audit M-6).
            fetch_dist = float(st.session_state.box_size_m) + 300.0
            from RF_prop_sim.coverage_engine import buildings_fetch_needed
            stale = buildings_fetch_needed(st.session_state.buildings_meta,
                                           st.session_state.map_center[0],
                                           st.session_state.map_center[1],
                                           fetch_dist) \
                if st.session_state.buildings_gdf is not None else True
            if st.session_state.buildings_gdf is None or stale:
                try:
                    gdf_new = download_buildings(
                        st.session_state.map_center[0],
                        st.session_state.map_center[1],
                        dist=int(fetch_dist),
                    )
                    if gdf_new is not None:
                        st.session_state.buildings_gdf = gdf_new
                        st.session_state.buildings_meta = {
                            "lat": st.session_state.map_center[0],
                            "lng": st.session_state.map_center[1],
                            "dist": fetch_dist,
                        }
                    elif st.session_state.buildings_gdf is None:
                        st.session_state.buildings_meta = None
                except Exception:
                    if st.session_state.buildings_gdf is None:
                        st.session_state.buildings_meta = None

            weather_kwargs = {
                "rain_rate_mmh": float(st.session_state.rain_rate_mmh),
                "relative_humidity": float(st.session_state.relative_humidity),
                "fog_liquid_water_density_gm3": float(st.session_state.fog_liquid_water_density_gm3),
                "temperature_c": float(st.session_state.temperature_c),
            }

            result = compute_coverage_result(
                antennas=transmitters,
                center_lat=st.session_state.map_center[0],
                center_lng=st.session_state.map_center[1],
                box_size_m=float(st.session_state.box_size_m),
                resolution_m=float(st.session_state.grid_res_m),
                model=st.session_state.def_model,
                combining=st.session_state.combining,
                buildings_gdf=st.session_state.buildings_gdf,
                **weather_kwargs,
            )
            st.session_state.coverage_result = result
            pts = result["points"] if result else []
            st.session_state.coverage_points = pts
            st.session_state.zone_stats = result["stats"] if result else zone_statistics([])
            st.session_state.receiver_reports = evaluate_receivers(
                receivers, transmitters,
                model=st.session_state.def_model,
                combining=st.session_state.combining,
                buildings_gdf=st.session_state.buildings_gdf,
                **weather_kwargs,
            )
            st.session_state.sim_ran = True
        st.rerun()

with col_clear:
    if st.button("🔄 Reset Map", use_container_width=True):
        st.session_state.coverage_points = []
        st.session_state.coverage_result = None
        st.session_state.zone_stats = {}
        st.session_state.sim_ran = False
        st.rerun()

if st.session_state.placing_mode:
    st.info("📍 **Placement mode active** — click anywhere on the map below to drop an antenna.")

# ── Model reliability banner (audit C1) ───────────────────────────────────────
_res = st.session_state.coverage_result
if _res and _res.get("warnings"):
    st.warning("⚠️ **Model reliability:** " + " · ".join(_res["warnings"]))

# ── Map ───────────────────────────────────────────────────────────────────────
st.markdown("### 🗺 Interactive Coverage Map")

def _map_fingerprint() -> str:
    """Content hash of everything that changes the map's VISUAL content.

    Deliberately EXCLUDES center/zoom: passive reruns (sidebar edits, etc.)
    must reuse the identical cached folium object so st_folium never remounts
    Leaflet — remounting is what snapped the view back to its stored center.
    Coverage enters only through its precomputed points_md5, keeping the
    fingerprint O(1) even for huge grids.
    """
    res = st.session_state.coverage_result
    payload = json.dumps({
        "antennas": st.session_state.antennas,
        "coverage_md5": res["points_md5"] if res else None,
        "receivers": st.session_state.receiver_reports,
        "has_buildings": st.session_state.buildings_gdf is not None,
        "placing": st.session_state.placing_mode,
        "box_size_m": float(st.session_state.box_size_m),
        "epoch": st.session_state.view_epoch,
    }, sort_keys=True, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()

_fp = _map_fingerprint()
_cache = st.session_state.folium_cache
if not (_cache and _cache["fingerprint"] == _fp):
    build_failed = False
    try:
        cached = _build_folium_map(
            center=st.session_state.map_center,
            zoom=st.session_state.map_zoom,
            antennas=st.session_state.antennas,
            coverage_points=st.session_state.coverage_points,
            buildings_gdf=st.session_state.buildings_gdf,
            receiver_reports=st.session_state.receiver_reports,
            box_size_m=st.session_state.box_size_m,
            coverage_result=st.session_state.coverage_result,
        )
    except Exception as e:
        # Never let a build failure silently blank the map: surface it and
        # degrade to a bare base map so the session stays interactive.
        # fingerprint=None makes the NEXT rerun retry the full build — caching
        # under _fp would make even a transient failure sticky forever.
        build_failed = True
        st.error(f"Map build failed: {e}")
        cached = folium.Map(
            location=st.session_state.map_center,
            zoom_start=st.session_state.map_zoom,
            tiles="CartoDB positron",
        )
    st.session_state.folium_cache = {"fingerprint": None if build_failed else _fp,
                                     "map": cached}

# streamlit_folium's render path DESTRUCTIVELY mutates the folium object it is
# given (_get_map_string renames element ids in place; see its source comment
# "_get_map_string alters the folium structure"). Re-rendering a previously
# rendered instance corrupts its generated JS -> blank, dead map. Hand the
# component a fresh deep copy each rerun: byte-identical content (no spurious
# remount -> Leaflet keeps pan/zoom), while mutations hit the throwaway copy.
folium_map = copy.deepcopy(st.session_state.folium_cache["map"])

map_data = st_folium(
    folium_map,
    width="100%",
    height=540,
    returned_objects=["last_clicked", "center", "zoom"],
    # Content-derived key: genuine content changes get a pristine widget slot
    # instead of recycling state across the library's internal hash rotations.
    key=f"main_map_{_fp[:8]}",
)

# Handle map click for antenna placement
if st.session_state.placing_mode and map_data and map_data.get("last_clicked"):
    click = map_data["last_clicked"]
    if click:
        idx = len(st.session_state.antennas) + 1
        new_ant = {
            "id": uuid4().hex[:8],
            "name": f"Antenna {idx}",
            "lat": click["lat"],
            "lng": click["lng"],
            "frequency_mhz": st.session_state.def_freq,
            "tx_power_dbm":  st.session_state.def_power,
            "gain_dbi":      st.session_state.def_gain,
            "height_m":      st.session_state.def_height,
            "nature":        st.session_state.def_nature,
        }
        st.session_state.antennas.append(new_ant)
        st.session_state.placing_mode = False
        # Clear old coverage when antennas change
        st.session_state.coverage_points = []
        st.session_state.coverage_result = None
        st.session_state.sim_ran = False
        st.success(f"✅ **{new_ant['name']}** placed at ({click['lat']:.5f}, {click['lng']:.5f})")
        st.rerun()

# Persist map view
if map_data:
    if map_data.get("center"):
        st.session_state.map_center = [map_data["center"]["lat"], map_data["center"]["lng"]]
    if map_data.get("zoom"):
        st.session_state.map_zoom = map_data["zoom"]

# ── Zone Statistics ───────────────────────────────────────────────────────────
if st.session_state.sim_ran and st.session_state.zone_stats \
        and st.session_state.zone_stats.get("total", 0) > 0:   # audit L-8
    st.markdown("---")
    st.markdown("###  Coverage Statistics")

    stats = st.session_state.zone_stats
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Grid Points", f"{stats['total']:,}")
    with m2:
        st.metric("🟢 Good Coverage", f"{stats['good_pct']}%",
                  help=f"{stats['good']:,} points ≥ {RSSI_GOOD} dBm")
    with m3:
        st.metric("🟠 Medium Coverage", f"{stats['medium_pct']}%",
                  help=f"{stats['medium']:,} points between {RSSI_MEDIUM} and {RSSI_GOOD} dBm")
    with m4:
        st.metric("🔴 Poor Coverage", f"{stats['bad_pct']}%",
                  help=f"{stats['bad']:,} points < {RSSI_MEDIUM} dBm")
    with m5:
        st.metric("Avg RSSI", f"{stats['avg_rssi_dbm']} dBm")

    # Zone bar
    good_w = stats["good_pct"]
    med_w  = stats["medium_pct"]
    bad_w  = stats["bad_pct"]
    st.markdown(
        f"""
        <div style="display:flex;height:12px;border-radius:6px;overflow:hidden;margin-top:8px;">
            <div style="width:{good_w}%;background:#22c55e;"></div>
            <div style="width:{med_w}%;background:#f97316;"></div>
            <div style="width:{bad_w}%;background:#ef4444;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;
                    font-size:11px;color:#94a3b8;margin-top:4px;">
            <span>🟢 {good_w}%</span>
            <span>🟠 {med_w}%</span>
            <span>🔴 {bad_w}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Export ────────────────────────────────────────────────────────────────
    def _coverage_geojson(points):
        # Audit L-7: describe the export with parameters captured at RUN time.
        res = st.session_state.coverage_result or {}
        run_params = res.get("run_params", {})
        """Build a GeoJSON FeatureCollection from coverage + receiver points."""
        features = []
        for pt in points:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [pt["lng"], pt["lat"]],
                },
                "properties": {
                    "feature_type": "grid_point",
                    "rssi_dbm": round(float(pt.get("rssi_dbm", -200.0)), 2),
                    "zone": pt.get("zone", "bad"),
                    "color": pt.get("color", "#ef4444"),
                    "best_antenna": pt.get("best_antenna", "None"),
                },
            })
        for rx in st.session_state.receiver_reports:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [rx["lng"], rx["lat"]],
                },
                "properties": {
                    "feature_type": "receiver",
                    "name": rx.get("name"),
                    "rssi_dbm": rx.get("rssi_dbm"),
                    "zone": rx.get("zone"),
                    "color": rx.get("color"),
                    "serving_antenna": rx.get("serving_antenna"),
                },
            })
        return json.dumps({
            "type": "FeatureCollection",
            "metadata": {
                # Audit L-7: prefer the parameters snapshotted AT RUN TIME so
                # exports describe their own data even after sidebar edits.
                "model": run_params.get("model", st.session_state.def_model),
                "combining": run_params.get("combining", st.session_state.combining),
                "grid_resolution_m": run_params.get(
                    "resolution_m", float(st.session_state.grid_res_m)),
                "analysis_area_m": {
                    "center_lat": float(st.session_state.map_center[0]),
                    "center_lng": float(st.session_state.map_center[1]),
                    "box_size_m": run_params.get(
                        "box_size_m", float(st.session_state.box_size_m)),
                },
                "weather": run_params.get("weather", {
                    "rain_rate_mmh": float(st.session_state.rain_rate_mmh),
                    "relative_humidity": float(st.session_state.relative_humidity),
                    "fog_liquid_water_density_gm3": float(st.session_state.fog_liquid_water_density_gm3),
                    "temperature_c": float(st.session_state.temperature_c),
                }),
                "buildings_used": bool(run_params.get(
                    "buildings_used", st.session_state.buildings_gdf is not None)),
                "antennas": [
                    {k: ant.get(k) for k in
                     ("name", "lat", "lng", "frequency_mhz", "tx_power_dbm",
                      "gain_dbi", "height_m", "nature")}
                    for ant in st.session_state.antennas
                ],
            },
            "features": features,
        }, indent=1)

    if st.session_state.coverage_points:
        gj_payload = _coverage_geojson(st.session_state.coverage_points)
        st.download_button(
            "⬇ Export Coverage (GeoJSON)",
            data=gj_payload,
            file_name="rf_coverage.geojson",
            mime="application/geo+json",
            use_container_width=True,
        )

# ── Per-Antenna Results ───────────────────────────────────────────────────────
if st.session_state.sim_ran and st.session_state.antennas:
    st.markdown("---")
    st.markdown("###  Per-Antenna Summary")

    cols = st.columns(min(len(st.session_state.antennas), 4))
    for i, ant in enumerate(st.session_state.antennas):
        with cols[i % len(cols)]:
            # Compute single-point FSPL to map center as quick reference
            import math
            d_km = max(0.001, math.sqrt(
                ((ant["lat"] - st.session_state.map_center[0]) * 111.32) ** 2 +
                ((ant["lng"] - st.session_state.map_center[1]) *
                 111.32 * math.cos(math.radians(ant["lat"]))) ** 2
            ))
            if d_km < 0.001:
                d_km = 0.001
            fspl = 32.44 + 20 * math.log10(d_km) + 20 * math.log10(ant["frequency_mhz"])
            rssi = ant["tx_power_dbm"] + ant["gain_dbi"] - fspl
            zone_emoji = "🟢" if rssi >= RSSI_GOOD else ("🟠" if rssi >= RSSI_MEDIUM else "🔴")

            st.markdown(
                f"""
                <div class="ant-card">
                  <b style="color:#a5b4fc;">{zone_emoji} {ant['name']}</b><br>
                  <span style="color:#94a3b8;font-size:12px;">
                     {ant['lat']:.4f}, {ant['lng']:.4f}<br>
                     {ant['frequency_mhz']} MHz | {ant['tx_power_dbm']} dBm<br>
                    RSSI @ centre: <b style="color:#e2e8f0;">{rssi:.1f} dBm</b>
                  </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── Receiver Report ───────────────────────────────────────────────────────────
if st.session_state.sim_ran and st.session_state.receiver_reports:
    st.markdown("---")
    st.markdown("###  Receiver Report")
    st.caption("Combined received signal at each receiver (power superposition across transmitters)")
    rows = []
    for rx in st.session_state.receiver_reports:
        emoji = "🟢" if rx["zone"] == "good" else ("🟠" if rx["zone"] == "medium" else "🔴")
        rows.append({
            "Receiver": f"{emoji} {rx['name']}",
            "RSSI (dBm)": rx["rssi_dbm"],
            "Zone": rx["zone"].capitalize(),
            "Serving TX": rx["serving_antenna"] if rx["covered"] else "— signal too weak —",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

# ── About ─────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ About & Model Reference"):
    st.markdown("""
    **RF Coverage Simulator** — built on top of `RF_prop_sim`.

    | Model | Description |
    |-------|-------------|
    | **FSPL** | Free Space Path Loss — ideal line-of-sight baseline |
    | **CI**   | Close-In reference model — urban/dense environments |
    | **ITM**  | Irregular Terrain Model (Longley-Rice) — hilly/rural terrain |
    | **Rain** | ITU-R P.838 rain attenuation |
    | **Gas**  | Atmospheric gas attenuation |
    | **Fog**  | ITU-R P.840 fog/cloud attenuation |

    **Coverage thresholds**
    - 🟢 **Good** → RSSI ≥ −80 dBm  
    - 🟠 **Medium** → −95 dBm ≤ RSSI < −80 dBm  
    - 🔴 **Bad** → RSSI < −95 dBm

    **Physics applied per link**
    - Base propagation loss (selected model; weather models include an FSPL base)
    - Terrain diffraction from real SRTM elevation profiles (4/3-earth bulge,
      knife-edge loss) — ITM receives full terrain profiles natively
    - Building wall losses from OSM footprints (+12 dB first wall, +6 dB each
      additional; +15 dB indoor) — weather is skipped for indoor points
    - **Superposition**: overlapping transmitters add in the power domain
      (`10·log₁₀ Σ 10^(RSSI/10)`), or classic best-server selection if toggled
    """)


if __name__ == "__main__":
    # Streamlit apps are run via `streamlit run`, so this block is not used.
    pass