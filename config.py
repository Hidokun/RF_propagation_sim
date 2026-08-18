import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# External API endpoints and service URLs
API_URLS = {
    "GOOGLE_MAPS_GEOCODE": "https://maps.googleapis.com/maps/api/geocode/json",
    "SAMPLE_DEM": "https://raw.githubusercontent.com/rasterio/rasterio/master/test/data/elevation.tif",
}

# Default fallback values when external services are unavailable
DEFAULTS = {
    "FALLBACK_GEOCODE_LOCATION": {
        "lat": 33.5883,
        "lng": -7.61138,
        "formatted_address": "Casablanca, Morocco (Fallback)",
    }
}

# DEM default path
DEFAULT_DEM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dem")

def get_dem_path():
    """Retrieve the path for DEM files, creating it if it doesn't exist."""
    path = os.environ.get("DEM_PATH", DEFAULT_DEM_PATH)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def get_api_key(service_name: str):
    """Retrieve API key for the specified service."""
    keys = {
        "GOOGLE_MAPS": "GOOGLE_MAPS_API_KEY",
    }
    env_var = keys.get(service_name)
    if not env_var:
        return None
        
    val = os.environ.get(env_var)
    if not val and service_name == "GOOGLE_MAPS":
        import json
        key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "google_key.json")
        if os.path.exists(key_path):
            try:
                with open(key_path, 'r') as f:
                    data = json.load(f)
                    return data.get("GOOGLE_MAPS_API_KEY")
            except Exception:
                pass
    return val

def get_api_url(endpoint_name: str):
    """Retrieve URL for the specified API endpoint."""
    return API_URLS.get(endpoint_name)
