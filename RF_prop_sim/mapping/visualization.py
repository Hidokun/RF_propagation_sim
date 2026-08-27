import os


def create_coverage_map(
    lat,
    lng,
    buildings_gdf=None,
    path_loss_data=None,
    antennas=None,
    output_file="coverage_map.html",
):
    """
    Create a 2D interactive Folium map with buildings and an RF coverage overlay.

    :param lat: Center latitude
    :param lng: Center longitude
    :param buildings_gdf: GeoDataFrame containing building polygons (from OSMnx)
    :param path_loss_data: List of coverage dicts from coverage_engine.compute_coverage_grid
                           Each dict: {lat, lng, rssi_dbm, zone, color, best_antenna}
    :param antennas: List of antenna dicts {lat, lng, name} to draw numbered markers
    :param output_file: Output HTML file path
    """
    try:
        import folium
    except ImportError:
        print("WARNING: folium is not installed. Run 'pip install folium'.")
        return None

    print(f"Generating 2D Folium Map centered at ({lat}, {lng})...")
    m = folium.Map(location=[lat, lng], zoom_start=15, tiles="CartoDB positron")

    # ── Buildings ─────────────────────────────────────────────────────────────
    if buildings_gdf is not None and not buildings_gdf.empty:
        try:
            # Audit L-18: osmnx output can carry crs=None; treat that as
            # already-lat/lon instead of dropping every building.
            if buildings_gdf.crs is not None and buildings_gdf.crs != "EPSG:4326":
                buildings_gdf = buildings_gdf.to_crs(epsg=4326)

            style_function = lambda x: {
                "fillColor": "#1e293b",
                "color": "#334155",
                "weight": 1,
                "fillOpacity": 0.6,
            }
            folium.GeoJson(
                buildings_gdf,
                name="Buildings",
                style_function=style_function,
            ).add_to(m)
            print(f"Added {len(buildings_gdf)} buildings to the map.")
        except Exception as e:
            print(f"Error adding buildings to map: {e}")

    # ── Coverage heat-point overlay ────────────────────────────────────────────
    if path_loss_data:
        cov_layer = folium.FeatureGroup(name="Coverage Zones", show=True)
        for point in path_loss_data:
            folium.CircleMarker(
                location=[point["lat"], point["lng"]],
                radius=5,
                color=point.get("color", "#6b7280"),
                fill=True,
                fill_color=point.get("color", "#6b7280"),
                fill_opacity=0.55,
                weight=0,
                popup=folium.Popup(
                    f"<b>RSSI:</b> {point['rssi_dbm']} dBm<br>"
                    f"<b>Zone:</b> {point['zone'].capitalize()}<br>"
                    f"<b>Best:</b> {point.get('best_antenna', '—')}",
                    max_width=180,
                ),
                tooltip=f"{point['rssi_dbm']} dBm ({point['zone']})",
            ).add_to(cov_layer)
        cov_layer.add_to(m)
        print(f"Added {len(path_loss_data)} coverage points to the map.")

    # ── Antenna markers ────────────────────────────────────────────────────────
    antenna_list = antennas or []
    if not antenna_list:
        # Fall back to a single centre marker
        antenna_list = [{"lat": lat, "lng": lng, "name": "Transmitter"}]

    colors = ["red", "blue", "purple", "orange", "darkred", "lightred",
              "beige", "darkblue", "darkgreen", "cadetblue"]

    ant_layer = folium.FeatureGroup(name="Antennas", show=True)
    for idx, ant in enumerate(antenna_list):
        a_lat = ant.get("lat") or lat
        a_lng = ant.get("lng") or lng
        name = ant.get("name", f"Antenna {idx + 1}")
        color = colors[idx % len(colors)]
        folium.Marker(
            [a_lat, a_lng],
            popup=folium.Popup(
                f"<b>{name}</b><br>"
                f"Freq: {ant.get('frequency_mhz', '—')} MHz<br>"
                f"Power: {ant.get('tx_power_dbm', '—')} dBm<br>"
                f"Gain: {ant.get('gain_dbi', '—')} dBi<br>"
                f"Height: {ant.get('height_m', '—')} m",
                max_width=200,
            ),
            tooltip=name,
            icon=folium.Icon(color=color, icon="signal", prefix="fa"),
        ).add_to(ant_layer)
    ant_layer.add_to(m)

    # ── Legend ─────────────────────────────────────────────────────────────────
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background: rgba(15,23,42,0.92); border-radius: 10px;
        padding: 12px 18px; box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        font-family: 'Inter', sans-serif; color: #f1f5f9;
        border: 1px solid rgba(255,255,255,0.1);">
        <p style="margin:0 0 8px 0; font-weight:700; font-size:13px; color:#94a3b8;">
            📶 Signal Coverage
        </p>
        <div style="display:flex;align-items:center;margin-bottom:5px;">
            <span style="width:14px;height:14px;border-radius:50%;background:#22c55e;
                         display:inline-block;margin-right:8px;flex-shrink:0;"></span>
            <span style="font-size:12px;">Good  (&ge; &minus;80 dBm)</span>
        </div>
        <div style="display:flex;align-items:center;margin-bottom:5px;">
            <span style="width:14px;height:14px;border-radius:50%;background:#f97316;
                         display:inline-block;margin-right:8px;flex-shrink:0;"></span>
            <span style="font-size:12px;">Medium (&minus;80 to &minus;95 dBm)</span>
        </div>
        <div style="display:flex;align-items:center;">
            <span style="width:14px;height:14px;border-radius:50%;background:#ef4444;
                         display:inline-block;margin-right:8px;flex-shrink:0;"></span>
            <span style="font-size:12px;">Bad  (&lt; &minus;95 dBm)</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # ── Layer control ──────────────────────────────────────────────────────────
    folium.LayerControl().add_to(m)

    m.save(output_file)
    print(f"Map successfully saved to {output_file}")
    return output_file


def render_3d_scene(scene_name="casablanca", tx_pos=None, rx_pos=None):
    """
    Use Sionna RT to render a 3D view of the scene.

    :param scene_name: The name of the built-in scene or path to an XML file.
    :param tx_pos: Transmitter position [x, y, z] (metres)
    :param rx_pos: Receiver position [x, y, z] (metres)
    """
    if tx_pos is None:
        tx_pos = [0, 0, 30]
    if rx_pos is None:
        rx_pos = [50, 50, 1.5]

    print("Generating 3D Visualization using Sionna RT...")
    try:
        import sionna.rt
        from sionna.rt import load_scene, Camera, Transmitter, Receiver

        # Audit M-12: be honest about scene substitution — silently rendering
        # simple_street_canyon while the log claims another city misleads.
        requested = getattr(sionna.rt.scene, scene_name, None)
        if requested is None:
            print(f"NOTE: Sionna has no builtin scene '{scene_name}'; "
                  f"falling back to 'simple_street_canyon'.")
            scene_name = "simple_street_canyon"
        scene = load_scene(getattr(sionna.rt.scene, scene_name))

        tx = Transmitter(name="tx", position=tx_pos, display_radius=2)
        scene.add(tx)

        rx = Receiver(name="rx", position=rx_pos, display_radius=2)
        scene.add(rx)

        tx.look_at(rx)

        my_cam = Camera(position=[-100, 100, 100], look_at=rx_pos)

        output_file = "scene_3d_render.png"
        scene.render_to_file(camera=my_cam, filename=output_file, resolution=[650, 500])
        print(f"3D Scene successfully rendered and saved to {output_file}")

    except ImportError:
        print("WARNING: sionna or mitsuba is not installed. Cannot render 3D scene.")
    except Exception as e:
        print(f"Error rendering 3D scene: {e}")
