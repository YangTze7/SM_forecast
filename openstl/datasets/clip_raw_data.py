import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
from shapely.geometry import box
import os
from glob import glob
# Define the longitude and latitude extent
lon_min, lat_min, lon_max, lat_max = [73.5005,18.250,135.000,53.750]  # Replace with your values

# Create a bounding box geometry
bbox = box(lon_min, lat_min, lon_max, lat_max)
data_dir = "/home/gldas/gldas_vars"
def load_tiff_path(data_dir,var_name):

    paths = glob(os.path.join(data_dir,var_name,'*.tif'))
    paths = sorted(paths)
  
    return paths
# Specify the input and output file paths
var_name = "SWdown_f_tavg"
paths = load_tiff_path(data_dir,var_name)
paths = sorted(paths)
for input_path in paths:

    input_raster_path = input_path

    output_raster_path = input_raster_path.replace('gldas_vars','gldas_vars_clipped')

    # Open the input raster file
    with rasterio.open(input_raster_path) as src:
        # Crop the raster using the bounding box
        out_image, out_transform = mask(src, [bbox], crop=True)
        out_meta = src.meta.copy()

        # Update metadata with new dimensions and transform
        out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform})

        # Save the clipped raster
        with rasterio.open(output_raster_path, "w", **out_meta) as dest:
            dest.write(out_image)

    print(output_raster_path)
