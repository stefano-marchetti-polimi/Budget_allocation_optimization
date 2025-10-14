import rasterio as rio
from pathlib import Path
from pygeoflood import pyGeoFlood
import pandas as pd
import numpy as np

pgf = pyGeoFlood(dem_path="data/houston_example_DEM_30m.tif")

# UTM 15N Easting, Northing of a grid cell in the ocean
ocean_E, ocean_N = 317540, 3272260

ocean_pixel = (ocean_E, ocean_N)  # map coords (E, N)
points = {
    "P1": (415337.819, 3321791.508),
    "P2": (471235.458, 3349033.547),
    "P3": (313798.678, 3292636.851),
    "P4": (429265.789,	3347691.776),
    "P5": (295005.562,	3302607.527),
    "P6": (307580.921,	3264174.781),
    "P7": (312418.912,	3251429.175),
    "P8": (265455.097,	3209409.708),
    "P9": (276712.4,	3242285.64),
}  # map coords (E, N), same CRS as the raster
inputs = np.linspace(0, 8, 2)  # meters
results = {}  # {depth: {point_name: value}}
Path("outputs").mkdir(parents=True, exist_ok=True)

for idx, val in enumerate(inputs, start=1):
    pgf.c_hand(ocean_coords=ocean_pixel, gage_el=val)

    # Sample raster values at all requested points for this hazard depth
    with rio.open(pgf.coastal_inundation_path) as ds:
        band1 = ds.read(1, masked=True)  # read only band 1
        for name, (P_E, P_N) in points.items():
            row, col = ds.index(P_E, P_N)  # map coords -> pixel indices
            value = band1[row, col]
            # Convert masked values to NaN and scalars to Python floats
            if np.ma.is_masked(value):
                v = float("nan")
            else:
                v = float(np.asarray(value))
            results.setdefault(val, {})[name] = v

# Convert results to a DataFrame: rows = points, columns = hazard depths (meters)
df = (
    pd.DataFrame(results)  # columns by depth, index by point name
    .sort_index(axis=1)    # sort columns by increasing depth
    .reindex(index=sorted(points.keys()))
)
out_path = Path("outputs/coastal_inundation_samples.csv")
#df.to_csv(out_path, float_format="%.6f")
print("Saved samples to:", out_path)
print(df)