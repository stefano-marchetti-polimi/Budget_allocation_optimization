from pathlib import Path
from pygeoflood import pyGeoFlood

pgf = pyGeoFlood(dem_path="data/houston_example_DEM_30m.tif")

"""### Show DEM with ocean pixel location"""

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio

from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.features import geometry_mask
from rasterio.plot import plotting_extent
from skimage.measure import label


def plot_raster(raster=None, profile=None, label=None, **kwargs):
    fig, ax = plt.subplots(dpi=200)

    # show inundation map
    im = ax.imshow(
        raster,
        extent=plotting_extent(raster, profile["transform"]),
        zorder=2,
        **kwargs,
    )

    # add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    fig.colorbar(im, cax=cax, label=label)

    # add basemap
    cx.add_basemap(
        ax,
        crs=profile["crs"],
        source=cx.providers.Esri.WorldImagery,
        zoom=10,
        attribution_size=2,
        zorder=1,
    )

    # add scalebar
    ax.add_artist(ScaleBar(1, box_alpha=0, location="lower right", color="white"))

    # add north arrow
    x, y, arrow_length = 0.9, 0.3, 0.15
    ax.annotate(
        "N",
        color="white",
        xy=(x, y),
        xytext=(x, y - arrow_length),
        arrowprops=dict(facecolor="white", edgecolor="white", width=5, headwidth=15),
        ha="center",
        va="center",
        fontsize=10,
        xycoords=ax.transAxes,
    )

    return fig, ax

# 30m DEM of houston to use as example
with rio.open(pgf.dem_path) as ds:
    dem = ds.read(1)
    dem_profile = ds.profile
    dem[dem == dem_profile["nodata"]] = np.nan

# UTM 15N Easting, Northing of a grid cell in the ocean
ocean_E, ocean_N = 317540, 3272260

fig, ax = plot_raster(
    raster=dem,
    profile=dem_profile,
    label="Elevation [m] NAVD88",
    interpolation="nearest",
    vmax=30,
    vmin=0,
    cmap="terrain",
)

ax.plot(ocean_E, ocean_N, "ro", label="Ocean Pixel")

# add labels
ax.legend(loc="upper left")
ax.set(
    title="USGS 1 arc-second DEM",
    xlabel="UTM 15N Easting [m]",
    ylabel="UTM 15N Northing [m]",
)

plt.show()

"""### run c-HAND"""

ocean_pixel = (ocean_E, ocean_N)
ike_gage = 0  # meters NAVD88
pgf.c_hand(ocean_coords=ocean_pixel, gage_el=ike_gage)

"""### Crop out coastline and plot coastal inundation map"""

# read coastal inundation map
with rio.open(pgf.coastal_inundation_path) as ds:
    inun_ike = ds.read(1)
    ike_profile = ds.profile

# read geojson or shapefile of domain with coastline
aoi_coast = gpd.read_file(Path("data", "aoi_coastline.geojson"))
aoi_coast_mask = geometry_mask(
    aoi_coast.geometry,
    inun_ike.shape,
    dem_profile["transform"],
)
# crop array to coastline
inun_ike[aoi_coast_mask == True] = np.nan
inun_ike[inun_ike == 0] = np.nan

fig, ax = plot_raster(
    raster=inun_ike,
    profile=ike_profile,
    label="Inundation Depth [m]",
    interpolation="nearest",
    cmap="Blues",
    vmax=ike_gage,
    vmin=0,
)

# Plot specified coordinates as red circles ("Components")
Easting = [
    415337.819, 471235.458, 313798.678, 244502.227, 257274.5, 220212.678, 255746.812,
    429265.789, 379554.557, 306767.939, 293518.032, 289480.745, 284106.364, 295005.562,
    298404.983, 307580.921, 312418.912, 234973.85, 265455.097, 408175.647, 468685.461
]
Northing = [
    3321791.508, 3349033.547, 3292636.851, 3263735.489, 3370348.078, 3388058.662, 3315018.623,
    3347691.776, 3353114.03, 3366966.79, 3288936.949, 3290269.241, 3289723.117, 3302607.527,
    3323055.016, 3264174.781, 3251429.175, 3217436.932, 3209409.708, 3306805.899, 3293817.284
]

ax.scatter(Easting, Northing, c='red', marker='o', s=20, label='Components', zorder=3)

# === Diagnostics: check which points fall within the raster extent ===
xmin, xmax, ymin, ymax = plotting_extent(inun_ike, ike_profile["transform"])
inside_flags = []
print("\n[Diagnostics] Component points inside raster extent:")
for idx, (e, n) in enumerate(zip(Easting, Northing), start=1):
    inside = (xmin <= e <= xmax) and (ymin <= n <= ymax)
    inside_flags.append(inside)
    print(f"  #{idx:02d}: E={e:.3f}, N={n:.3f} -> {'INSIDE' if inside else 'OUTSIDE'}")
    # Label each point on the map for visual counting
    ax.text(e, n, str(idx), fontsize=6, color='yellow', ha='center', va='bottom', zorder=4)

# If any points are outside, mention it in the title suffix
if not all(inside_flags):
    n_inside = sum(inside_flags)
    ax.set_title(f"Hazard Map (showing {n_inside}/21 points within extent)")

ax.legend(loc="upper left")

# add labels
ax.set(
    title="Hazard Map",
    xlabel="UTM 15N Easting [m]",
    ylabel="UTM 15N Northing [m]",
)

plt.show()

# save figure
fig.savefig(Path("data", "inun_ike.png"), dpi=200, bbox_inches="tight")

# === Additional plot: baseline background with zero inundation ===
fig_bg, ax_bg = plot_raster(
    raster=inun_ike,
    profile=ike_profile,
    label="Inundation Depth [m]",
    interpolation="nearest",
    cmap="Blues",
    vmax=ike_gage,
    vmin=0,
)
ax_bg.set(
    title="Baseline Map (0 m Inundation)",
    xlabel="UTM 15N Easting [m]",
    ylabel="UTM 15N Northing [m]",
)
plt.show()

fig_bg.savefig(Path("data", "inun_zero_depth.png"), dpi=200, bbox_inches="tight")

# === Additional plot: color-coded assets with numbering ===
fig2, ax2 = plot_raster(
    raster=inun_ike,
    profile=ike_profile,
    label="Inundation Depth [m]",
    interpolation="nearest",
    cmap="Blues",
    vmax=ike_gage,
    vmin=0,
)

# Coordinates (Easting, Northing) for the 20 assets
Easting2 = [
    244502.227, 472219.745, 377268.615, 282964.652, 329137.333,
    357346.492, 356656.517, 321434.383, 280600.329, 264056.064,
    194397.422, 211431.374, 234703.247, 231466.175, 188893.778,
    205442.32, 215477.073, 352215.605, 331064.617, 445207.189
]
Northing2 = [
    3263735.489, 3350333.29, 3352529.866, 3354562.856, 3394254.312,
    3413398.73, 3320856.71, 3315671.476, 3239249.39, 3235743.32,
    3229708.654, 3232827.119, 3256488.038, 3230851.796, 3273613.229,
    3235021.604, 3196837.539, 3192941.72, 3137635.929, 3251671.531
]

# Asset types, in order (parallel to coordinates)
asset_types = [
    "Coal Power Station", "Coal Power Station",
    "Solar Power Station", "Solar Power Station", "Solar Power Station",
    "Solar Power Station", "Solar Power Station", "Solar Power Station",
    "Solar Power Station", "Solar Power Station", "Solar Power Station",
    "Solar Power Station", "Solar Power Station", "Solar Power Station",
    "Solar Power Station", "Wind Power Station", "Wind Power Station",
    "Offshore Wind Farm", "Offshore Wind Farm", "Offshore Wind Farm"
]

# Color map per asset type
color_map = {
    "Coal Power Station": "black",
    "Solar Power Station": "yellow",
    "Wind Power Station": "lime",
    "Offshore Wind Farm": "cyan",
}

# Plot points color-coded by asset type
colors2 = [color_map[t] for t in asset_types]
ax2.scatter(Easting2, Northing2, c=colors2, marker='o', s=28, edgecolors='k', linewidths=0.5, zorder=3)

# Number labels (1..20) next to each point
for idx, (e, n) in enumerate(zip(Easting2, Northing2), start=1):
    ax2.text(e, n, str(idx), fontsize=7, color='red', ha='center', va='bottom', zorder=4)

# Legend using proxy artists
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Coal Power Station', markerfacecolor=color_map['Coal Power Station'], markeredgecolor='k', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Solar Power Station', markerfacecolor=color_map['Solar Power Station'], markeredgecolor='k', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Wind Power Station', markerfacecolor=color_map['Wind Power Station'], markeredgecolor='k', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Offshore Wind Farm', markerfacecolor=color_map['Offshore Wind Farm'], markeredgecolor='k', markersize=8),
]
ax2.legend(handles=legend_elements, loc='lower right', title='Asset Type')

# Add labels
ax2.set(
    title="Assets Map (numbered, color-coded)",
    xlabel="UTM 15N Easting [m]",
    ylabel="UTM 15N Northing [m]",
)

plt.show()

# Save second figure
fig2.savefig(Path("data", "inun_assets.png"), dpi=200, bbox_inches="tight")
