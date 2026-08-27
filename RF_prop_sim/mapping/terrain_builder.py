import os
import sys

# Add the project root directory to the path so we can import config
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def download_buildings(lat, lng, dist=500):
    """
    Download building footprints using OpenStreetMap (OSMnx).
    
    :param lat: Latitude
    :param lng: Longitude
    :param dist: Distance in meters from the center point to download
    :return: GeoDataFrame containing building polygons, or None if it fails
    """
    try:
        import osmnx as ox
        print(f"Downloading buildings around ({lat}, {lng}) with radius {dist}m...")
        tags = {'building': True}
        gdf = ox.features_from_point((lat, lng), tags, dist=dist)
        
        if gdf.empty:
            print("No buildings found in this area.")
            return None
            
        return gdf
    except ImportError:
        print("WARNING: osmnx is not installed. Run 'pip install osmnx'.")
        return None
    except Exception as e:
        print(f"Error downloading buildings: {e}")
        return None

from .dem_provider import DemProvider
import config

# Basenames treated as synthetic fallbacks rather than real terrain data
_SYNTHETIC_DEM_PREFIXES = ("sample_dem",)


def resolve_dem_path(prefer_real=True):
    """
    Resolve a usable DEM .tif path using the documented preference order:
      1. Any real tile in data/dem (excluding synthetic fallbacks)
      2. The bundled synthetic sample_dem.tif
      3. None if nothing exists (caller decides whether to generate)

    Returns an absolute path string or None.
    """
    dem_dir = config.get_dem_path()
    import glob
    all_tifs = sorted(glob.glob(os.path.join(dem_dir, "*.tif")))
    if not all_tifs:
        return None
    real_tifs = [t for t in all_tifs
                 if not os.path.basename(t).startswith(_SYNTHETIC_DEM_PREFIXES)]
    if prefer_real and real_tifs:
        return real_tifs[0]
    return all_tifs[0]


def get_elevation(lat, lng, dem_path=None):
    """
    Return elevation for a given latitude and longitude using a local DEM file.

    Preference order:
      1. Explicitly provided dem_path
      2. Any real DEM tile in data/dem (excluding synthetic fallbacks)
      3. The bundled synthetic sample (loudly announced)

    :param lat: Latitude in decimal degrees
    :param lng: Longitude in decimal degrees
    :param dem_path: Optional local DEM file path. If omitted, reads DEM_PATH from config.
    :return: Elevation in meters
    """
    if dem_path is None:
        resolved = resolve_dem_path()
        if resolved is None:
            print("WARNING: No DEM file found in data/dem. Downloading sample DEM...")
            from .dem_processor import download_sample_dem
            dem_dir = config.get_dem_path()
            dem_path = os.path.join(dem_dir, "sample_dem.tif")
            download_sample_dem(dem_path)
            print("WARNING: Synthetic DEM generated. Elevations are NOT real terrain.")
        else:
            dem_path = resolved
            if os.path.basename(dem_path).startswith(_SYNTHETIC_DEM_PREFIXES):
                print("WARNING: Using SYNTHETIC sample_dem.tif (elevations are fake). "
                      "Run scripts/fetch_srtm_dem.py to obtain real terrain data.")
            
    if not dem_path or not os.path.exists(dem_path):
        print("WARNING: No valid DEM path resolved. Returning 0.0 elevation.")
        return 0.0

    try:
        with DemProvider(dem_path) as dem:
            return dem.get_elevation(lat, lng)
    except ImportError:
        print("WARNING: rasterio is not installed. Install with 'pip install rasterio'.")
        return 0.0
    except Exception as e:
        print(f"Error reading elevation from DEM: {e}")
        return 0.0
