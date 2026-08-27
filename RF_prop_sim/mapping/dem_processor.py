import os
import sys
import urllib.request

# Add the project root directory to the path so we can import config
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def download_sample_dem(output_path="sample_dem.tif"):
    """
    Downloads a sample SRTM DEM (.tif) file to use with PyVista.
    We use an open data source for a small area.
    """
    # URL to a small sample GeoTIFF (public domain)
    # Since dynamic fetching requires complex auth for NASA Earthdata,
    # we download a widely available public sample DEM
    from config import get_api_url

    url = get_api_url("SAMPLE_DEM")
    
    print(f"Downloading sample DEM from {url}...")
    try:
        # Audit L-11: urlretrieve has no timeout and can hang forever.
        import shutil
        with urllib.request.urlopen(url, timeout=60) as resp, \
                open(output_path, "wb") as out_f:
            shutil.copyfileobj(resp, out_f)
        print(f"Sample DEM downloaded to {output_path}")
        return output_path
    except Exception as e:
        print(f"Failed to download DEM: {e}")
        print("Generating a synthetic sample DEM...")
        import rasterio
        from rasterio.transform import from_origin
        import numpy as np
        
        # Create a 100x100 DEM centered around Casablanca
        data = np.linspace(10, 50, 10000).reshape((100, 100)).astype(np.float32)
        transform = from_origin(-7.61138 - 0.05, 33.58831 + 0.05, 0.001, 0.001)
        
        try:
            with rasterio.open(
                output_path, 'w', driver='GTiff',
                height=data.shape[0], width=data.shape[1],
                count=1, dtype=data.dtype,
                crs='+proj=latlong', transform=transform
            ) as dst:
                dst.write(data, 1)
            print(f"Synthetic DEM created at {output_path}")
            return output_path
        except Exception as e2:
            print(f"Failed to create synthetic DEM: {e2}")
            return None

def render_dem_3d(dem_file_path):
    """
    Load a DEM using rioxarray and visualize in 3D using PyVista.
    As instructed in Dem_pyvista.txt.
    """
    try:
        import pyvista as pv
        import rioxarray as riox
        import numpy as np
    except ImportError:
        print("WARNING: pyvista or rioxarray not installed. Run 'pip install pyvista rioxarray'.")
        return
        
    print(f"Loading DEM data from {dem_file_path}...")
    try:
        data = riox.open_rasterio(dem_file_path)
        data = data[0]
        
        # Save the raster data as an array
        values = np.asarray(data)
        
        # Create a mesh grid
        x, y = np.meshgrid(data['x'], data['y'])
        
        # Set the z values and create a StructuredGrid
        z = np.zeros_like(x)
        mesh = pv.StructuredGrid(x, y, z)
        
        # Assign Elevation Values
        mesh["Elevation"] = values.ravel(order='F')
        
        # Warp the mesh by scalar
        # Adjust factor based on coordinate system; since it's lat/lon, elevation needs heavy scaling down
        topo = mesh.warp_by_scalar(scalars="Elevation", factor=0.000015)
        
        # Plot the elevation map
        print("Opening 3D PyVista window (Close the window to continue)...")
        p = pv.Plotter()
        p.add_mesh(mesh=topo, scalars=topo["Elevation"], cmap='terrain')
        p.show_grid(color='black')
        p.set_background(color='white')
        p.show(cpos="xy")
        
    except Exception as e:
        print(f"Error visualizing DEM: {e}")
