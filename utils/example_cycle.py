import rasterio as rio
from pathlib import Path
from pygeoflood import pyGeoFlood

pgf = pyGeoFlood(dem_path="data/houston_example_DEM_30m.tif")

# UTM 15N Easting, Northing of a grid cell in the ocean
ocean_E, ocean_N = 317540, 3272260

ocean_pixel = (ocean_E, ocean_N)  # map coords (E, N)
P_E, P_N = 450000, 3300000        # the point you want to sample (map coords)
inputs = [2, 3, 4, 5]
results = {}

for idx, val in enumerate(inputs, start=1):
    pgf.c_hand(ocean_coords=ocean_pixel, gage_el=val)

    with rio.open(pgf.coastal_inundation_path) as ds:
        # If P_E,P_N are in the same CRS as ds (check ds.crs)
        row, col = ds.index(P_E, P_N)        # map coords -> pixel indices
        band1 = ds.read(1, masked=True)      # read only band 1
        results[idx] = band1[row, col].item()  # store scalar (float)

print(results)