# RF Simulator UI - Streamlit Implementation

This directory contains the Streamlit-based user interface for the RF Propagation Simulator.

## Overview

The UI has been migrated from a React-based architecture to Streamlit for:
- Faster development and iteration
- Direct integration with Python backend
- Reduced complexity in frontend/backend communication
- Built-in widgets optimized for scientific/technical applications

## Structure

```
UI/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies for UI
�└── README.md           # This file
```

## Key Features

- **Interactive Parameter Controls**: Sliders, dropdowns, checkboxes for all simulation parameters
- **File Upload Support**: Config and antenna file uploaders
- **Google Maps Integration**: API key handling for mapping features
- **Real-time Simulation**: Button-driven execution with progress indicators
- **Results Visualization**: Maps, metrics, and data display
- **Responsive Layout**: Sidebar for parameters, main area for results
- **Help System**: Expandable sections with tooltips and documentation

## Dependencies

All dependencies are listed in `requirements.txt` and are strictly contained within this UI directory:
- `streamlit`: Main framework
- `streamlit-components-v1`: For custom components if needed

## Usage

To run the Streamlit UI:

```bash
# From the UI directory:
streamlit run app.py

# Or from project root:
streamlit run UI/app.py
```

## Integration with Backend

The UI directly imports and uses:
- `RF_prop_sim.main.run_simulation()`: Core simulation function
- `RF_prop_sim.input_data_collection.ingestion.parse_simulation_config()`: Config parsing
- `RF_prop_sim.input_data_collection.ingestion.SimulationConfig`: Configuration dataclass

No changes were made to the backend code - the UI integrates with existing functionality.

## Configuration Parameters Supported

All parameters from the `SimulationConfig` dataclass are supported:
- Location: address, coordinates, area radius
- Propagation: model, frequency, distance, heights
- Atmospheric: rain rate, fog density, temperature, pressure, humidity
- Antenna: power, gain, coordinates, config file
- Advanced: optimization, DEM generation, model-specific parameters
- Output: output folder, Google Maps API key handling

## Customization

To modify the UI:
1. Edit `UI/app.py` to change layout, add/remove parameters, or modify styling
2. Update `UI/requirements.txt` if additional Python packages are needed
3. The UI will automatically reflect changes when saved (Streamlit hot-reload)

## Deployment

The UI can be deployed anywhere Streamlit is supported:
- Local development: `streamlit run UI/app.py`
- Cloud platforms: Streamlit Community Cloud, AWS, Azure, etc.
- Docker containers: Standard Streamlit Docker images work

Note: Keep all Streamlit-related files within this UI/ directory to maintain clean separation from backend code.