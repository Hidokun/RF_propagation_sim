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

def get_elevation(lat, lng, dem_path=None):
    """
    Return elevation for a given latitude and longitude using a local DEM file.

    :param lat: Latitude in decimal degrees
    :param lng: Longitude in decimal degrees
    :param dem_path: Optional local DEM file path. If omitted, reads DEM_PATH from config.
    :return: Elevation in meters
    """
    if dem_path is None:
        dem_dir = config.get_dem_path()
        # Find a .tif file in the directory
        import glob
        tif_files = glob.glob(os.path.join(dem_dir, "*.tif"))
        if tif_files:
            dem_path = tif_files[0]
        else:
            print("WARNING: No DEM file found in data/dem. Downloading sample DEM...")
            from .dem_processor import download_sample_dem
            dem_path = os.path.join(dem_dir, "sample_dem.tif")
            download_sample_dem(dem_path)
            
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
