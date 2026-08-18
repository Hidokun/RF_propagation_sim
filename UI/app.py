import streamlit as st
import sys
import os
import json
import glob

# Add the project root to the path so we can import RF_prop_sim modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from RF_prop_sim.main import run_simulation
from RF_prop_sim.input_data_collection.ingestion import parse_simulation_config, SimulationConfig

def load_sim_config():
    if st.session_state.sim_config_file and st.session_state.sim_config_file != "None":
        try:
            with open(st.session_state.sim_config_file, 'r') as f:
                data = json.load(f)
            if 'address' in data: st.session_state.address = data['address']
            if 'model' in data: st.session_state.model = data['model']
            if 'frequency_mhz' in data: st.session_state.frequency_mhz = float(data['frequency_mhz'])
            if 'distance_km' in data: st.session_state.distance_km = float(data['distance_km'])
            if 'tx_height_m' in data: st.session_state.tx_height_m = float(data['tx_height_m'])
            if 'rx_height_m' in data: st.session_state.rx_height_m = float(data['rx_height_m'])
            if 'area_radius_m' in data: st.session_state.area_radius_m = float(data['area_radius_m'])
            if 'tx_power_dbm' in data: st.session_state.tx_power_dbm = float(data['tx_power_dbm'])
            if 'antenna_gain_dbi' in data: st.session_state.gain_dbi = float(data['antenna_gain_dbi'])
            if 'center_lat' in data and 'center_lng' in data:
                st.session_state.use_custom_coordinates = True
                st.session_state.center_lat = float(data['center_lat'])
                st.session_state.center_lng = float(data['center_lng'])
        except Exception as e:
            st.sidebar.error(f"Error loading sim config: {e}")

def load_antenna_config():
    if st.session_state.ant_config_file and st.session_state.ant_config_file != "None":
        try:
            with open(st.session_state.ant_config_file, 'r') as f:
                data = json.load(f)
            if 'tx_power_dbm' in data: st.session_state.tx_power_dbm = float(data['tx_power_dbm'])
            if 'frequency_mhz' in data: st.session_state.frequency_mhz = float(data['frequency_mhz'])
            if 'height_m' in data: st.session_state.tx_height_m = float(data['height_m'])
            if 'gain_dbi' in data: st.session_state.gain_dbi = float(data['gain_dbi'])
        except Exception as e:
            st.sidebar.error(f"Error loading antenna config: {e}")

def init_session_state():
    defaults = {
        'address': "Casablanca, Morocco",
        'use_custom_coordinates': True,
        'center_lat': 33.58831,
        'center_lng': -7.61138,
        'area_radius_m': 500.0,
        'tx_power_dbm': 40.0,
        'frequency_mhz': 900.0,
        'tx_height_m': 30.0,
        'rx_height_m': 1.5,
        'gain_dbi': 12.0,
        'model': "fspl",
        'distance_km': 0.5
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def main():
    # Page configuration
    st.set_page_config(
        page_title="RF Propagation Simulator",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()

    st.title("RF Propagation Simulator")
    st.markdown("### Interactive Radio Frequency Coverage Analysis Tool")
    st.markdown("---")

    # Get json files
    examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    json_files = ["None"]
    if os.path.exists(examples_dir):
        json_files.extend(glob.glob(os.path.join(examples_dir, "*.json")))

    # Sidebar for input parameters
    with st.sidebar:
        st.header("Dataset Selection")
        st.selectbox(
            "Simulation Configuration",
            options=json_files,
            key="sim_config_file",
            on_change=load_sim_config,
            format_func=lambda x: os.path.basename(x) if x != "None" else "None"
        )
        st.selectbox(
            "Antenna Configuration",
            options=json_files,
            key="ant_config_file",
            on_change=load_antenna_config,
            format_func=lambda x: os.path.basename(x) if x != "None" else "None"
        )
        st.markdown("---")
        
        st.header("Simulation Parameters")
        
        # Location Section
        st.subheader("Location Settings")
        address = st.text_input(
            "Address or Location", 
            key="address",
            help="Enter an address or location name for geocoding"
        )
        
        # Advanced location options
        use_custom_coordinates = st.checkbox("Use Custom Coordinates", key="use_custom_coordinates")
        if use_custom_coordinates:
            col1, col2 = st.columns(2)
            with col1:
                center_lat = st.number_input(
                    "Latitude", 
                    key="center_lat",
                    format="%.6f",
                    help="Geographic latitude coordinate"
                )
            with col2:
                center_lng = st.number_input(
                    "Longitude", 
                    key="center_lng",
                    format="%.6f",
                    help="Geographic longitude coordinate"
                )
        else:
            center_lat = None
            center_lng = None
            
        area_radius_m = st.number_input(
            "Area Radius (m)", 
            min_value=50.0, 
            key="area_radius_m",
            help="Radius around location to simulate coverage"
        )
        
        st.markdown("---")
        
        # Antenna Section
        st.subheader("Antenna Configuration")
        tx_power_dbm = st.number_input(
            "Transmit Power (dBm)", 
            key="tx_power_dbm",
            help="Power level of the transmitted signal"
        )
        frequency_mhz = st.number_input(
            "Frequency (MHz)", 
            key="frequency_mhz",
            min_value=1.0,
            help="Frequency of the signal"
        )
        height_m = st.number_input(
            "Antenna Height (m)", 
            key="tx_height_m",
            help="Height of the antenna above ground"
        )
        gain_dbi = st.number_input(
            "Antenna Gain (dBi)", 
            key="gain_dbi",
            help="Gain of the antenna"
        )
        
        st.markdown("---")
        
        # Simulation Section
        st.subheader("Simulation Settings")
        model = st.selectbox(
            "Propagation Model",
            ["fspl", "rain", "ci", "itm", "sionna", "gas", "fog"],
            key="model",
            help="Select the propagation model to use"
        )
        distance_km = st.number_input(
            "Distance (km)", 
            key="distance_km",
            min_value=0.01,
            help="Distance for point-to-point calculation"
        )
        rx_height_m = st.number_input(
            "Receiver Height (m)", 
            key="rx_height_m",
            help="Height of receiver above ground"
        )
        
        st.markdown("---")
        
        # Run Simulation Button
        run_button = st.button(" Run Simulation", type="primary")

    # Main content area
    if run_button:
        with st.spinner("Running simulation..."):
            try:
                # Create a simple config object for now
                config_dict = {
                    "address": st.session_state.address,
                    "model": st.session_state.model,
                    "frequency_mhz": st.session_state.frequency_mhz,
                    "distance_km": st.session_state.distance_km,
                    "tx_height_m": st.session_state.tx_height_m,
                    "rx_height_m": st.session_state.rx_height_m,
                    "area_radius_m": st.session_state.area_radius_m,
                    "tx_power_dbm": st.session_state.tx_power_dbm,
                    "antenna_gain_dbi": st.session_state.gain_dbi
                }
                
                if st.session_state.ant_config_file and st.session_state.ant_config_file != "None":
                    config_dict["antenna_config_path"] = st.session_state.ant_config_file
                
                # Handle custom coordinates if provided and valid
                if (st.session_state.use_custom_coordinates and 
                    st.session_state.center_lat is not None and 
                    st.session_state.center_lng is not None and
                    # Additional validation: check that coordinates are not at default values
                    st.session_state.center_lat != 0.0 and 
                    st.session_state.center_lng != 0.0):
                    config_dict["center_lat"] = st.session_state.center_lat
                    config_dict["center_lng"] = st.session_state.center_lng
                
                # Save config to temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(config_dict, f)
                    config_path = f.name
                
                # Run simulation
                from RF_prop_sim.input_data_collection.ingestion import load_config_from_file
                from antenna_data.parser import parse_antenna_config
                
                cfg = load_config_from_file(config_path)
                antenna_cfg = None
                
                if st.session_state.ant_config_file and st.session_state.ant_config_file != "None":
                    if hasattr(cfg, 'antenna_config_path') and cfg.antenna_config_path:
                        try:
                            antenna_cfg = parse_antenna_config(cfg.antenna_config_path)
                        except Exception as e:
                            st.sidebar.error(f"Error loading antenna config: {e}")
                            # Continue with None antenna_cfg to let run_simulation handle it
                    else:
                        st.sidebar.warning("Antenna config file selected but no path found in configuration")
                    
                result = run_simulation(cfg, antenna_cfg=antenna_cfg)
                
                # Display results
                st.success("Simulation completed successfully!")
                
                # Results metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Path Loss", f"{result['path_loss_db']:.2f} dB")
                with col2:
                    st.metric("Model Used", result['model'].upper())
                with col3:
                    st.metric("Status", result['status'].upper())
                with col4:
                    st.metric("Elevation", f"{result.get('elevation_m', 0):.1f} m")
                
                # Display location info
                if 'location' in result:
                    loc = result['location']
                    lat = loc.get('lat')
                    lng = loc.get('lng')
                    formatted_address = loc.get('formatted_address')
                    
                    if lat is not None and lng is not None:
                        location_str = f"{formatted_address}" if formatted_address else f"{lat:.4f}, {lng:.4f}"
                    else:
                        location_str = formatted_address if formatted_address else "Location coordinates unavailable"
                    
                    st.info(f"**Location:** {location_str}")
                
                # Display map if available
                if 'map_file' in result and result['map_file']:
                    st.subheader("Coverage Map")
                    try:
                        with open(result['map_file'], 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=500)
                    except Exception as e:
                        st.warning(f"Could not display map: {e}")
                
                # Show detailed results in expandable section
                with st.expander("Detailed Results"):
                    st.json(result)
                    
            except Exception as e:
                st.error(f"Error running simulation: {str(e)}")
                st.exception(e)

    # Information section
    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    This RF Propagation Simulator demonstrates radio frequency signal propagation 
    using various models including:
    - **FSPL**: Free Space Path Loss
    - **Rain**: Rain Attenuation Model
    - **CI**: Close-in Path Loss Model
    - **ITM**: Irregular Terrain Model
    - **Sionna**: Ray Tracing-based Model
    - **Gas**: Gas Attenuation Model
    - **Fog**: Fog Attenuation Model
    
    Enter a location, configure antenna parameters, select a propagation model, 
    and run the simulation to see predicted signal strength and coverage.
    """)

if __name__ == "__main__":
    main()