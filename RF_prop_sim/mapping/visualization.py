import os

def create_coverage_map(lat, lng, buildings_gdf=None, path_loss_data=None, output_file="coverage_map.html"):
    """
    Create a 2D interactive Folium map with buildings and an RF propagation overlay.
    
    :param lat: Center latitude
    :param lng: Center longitude
    :param buildings_gdf: GeoDataFrame containing building polygons (from OSMnx)
    :param path_loss_data: Dictionary or list of path loss data to overlay (e.g., {'lat': x, 'lng': y, 'loss': z})
    :param output_file: Output HTML file name
    """
    try:
        import folium
    except ImportError:
        print("WARNING: folium is not installed. Run 'pip install folium'.")
        return None

    print(f"Generating 2D Folium Map centered at ({lat}, {lng})...")
    m = folium.Map(location=[lat, lng], zoom_start=15)
    
    # Add buildings if provided
    if buildings_gdf is not None and not buildings_gdf.empty:
        try:
            import geopandas as gpd
            # Ensure the CRS is EPSG:4326 for Folium
            if buildings_gdf.crs != "EPSG:4326":
                buildings_gdf = buildings_gdf.to_crs(epsg=4326)
                
            style_function = lambda x: {'fillColor': '#000000', 'color': '#000000', 'weight': 1, 'fillOpacity': 0.5}
            folium.GeoJson(
                buildings_gdf,
                name="Buildings",
                style_function=style_function
            ).add_to(m)
            print(f"Added {len(buildings_gdf)} buildings to the map.")
        except Exception as e:
            print(f"Error adding buildings to map: {e}")

    # Add a marker for the center/transmitter
    folium.Marker(
        [lat, lng], 
        popup="Transmitter Location", 
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)

    # Note: If path_loss_data is provided, we would add a HeatMap or colored markers here.
    # For now, it's a placeholder for the heat map logic.

    # Save to file
    m.save(output_file)
    print(f"Map successfully saved to {output_file}")
    return output_file

def render_3d_scene(scene_name="casablanca", tx_pos=[0,0,30], rx_pos=[50,50, 1.5]):
    """
    Use Sionna RT to render a 3D view of the scene.
    
    :param scene_name: The name of the built-in scene or path to an XML file.
    :param tx_pos: Transmitter position
    :param rx_pos: Receiver position
    """
    print("Generating 3D Visualization using Sionna RT...")
    try:
        import sionna.rt
        from sionna.rt import load_scene, Camera, Transmitter, Receiver
        
        scene = load_scene(getattr(sionna.rt.scene, scene_name, sionna.rt.scene.simple_street_canyon))
        
        tx = Transmitter(name="tx", position=tx_pos, display_radius=2)
        scene.add(tx)

        rx = Receiver(name="rx", position=rx_pos, display_radius=2)
        scene.add(rx)

        tx.look_at(rx)
        
        my_cam = Camera(position=[-100, 100, 100], look_at=rx_pos)
        
        # Render scene to file
        output_file = "scene_3d_render.png"
        scene.render_to_file(camera=my_cam, filename=output_file, resolution=[650, 500])
        print(f"3D Scene successfully rendered and saved to {output_file}")
        
    except ImportError:
        print("WARNING: sionna or mitsuba is not installed. Cannot render 3D scene.")
    except Exception as e:
        print(f"Error rendering 3D scene: {e}")
