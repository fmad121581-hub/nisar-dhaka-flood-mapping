"""
Figure 1 (and supplementary locator maps) -- real WorldCover context maps
for all four study sites, cropped from the actual ESA WorldCover 10m raw
tiles (not a schematic), with each site's exact analysis bounding box drawn
as an overlay rectangle. No GADM/geoBoundaries polygon file is present on
this disk, so no administrative boundary line is drawn -- the padded raster
crop itself provides surrounding geographic context instead.
"""
import os
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

# Figure titles are intentionally NOT drawn inside the image: Copernicus/NHESS
# requires the caption to be typeset text below the figure, so a title baked into
# the PNG would duplicate it. Captions live in the manuscript, not here.

_HOME = Path(os.environ["HOME"])
D_ROOT = _HOME / "mnt" / "d" / "NISAR_Dhaka_Flood"
WC_DIR = D_ROOT / "raw_data" / "worldcover"
FIG_DIR = D_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TILE_N21 = WC_DIR / "ESA_WorldCover_10m_2020_v100_N21E090_Map.tif"
TILE_N24 = WC_DIR / "ESA_WorldCover_10m_2020_v100_N24E090_Map.tif"

SITES = {
    "Demra":         {"bbox": (240673.90374, 2621657.94409, 246404.89660, 2630128.18446), "tile": TILE_N21, "pad": 1.5},
    "Uttar Khan":    {"bbox": (236582.74858659075, 2639492.080730549, 242585.99816525163, 2645681.772454119), "tile": TILE_N21, "pad": 1.5},
    "Hakaluki Haor": {"bbox": (398749.23, 2719084.57, 412351.52, 2735786.23), "tile": TILE_N24, "pad": 0.8},
    "Kaptai Lake":   {"bbox": (402869.91, 2485087.90, 433476.61, 2549171.39), "tile": TILE_N21, "pad": 0.6},
}

WC_COLORS = {
    10: ("Tree cover", "#006400"), 20: ("Shrubland", "#ffbb22"), 30: ("Grassland", "#ffff4c"),
    40: ("Cropland", "#f096ff"), 50: ("Built-up", "#fa0000"), 60: ("Bare/sparse", "#b4b4b4"),
    70: ("Snow/ice", "#f0f0f0"), 80: ("Permanent water", "#0064c8"), 90: ("Herbaceous wetland", "#0096a0"),
    95: ("Mangroves", "#00cf75"), 100: ("Moss/lichen", "#fae6a0"),
}
codes_sorted = sorted(WC_COLORS)
code_to_i = {c: i for i, c in enumerate(codes_sorted)}
cmap = ListedColormap([WC_COLORS[c][1] for c in codes_sorted])


def code_to_index(arr):
    idx = np.zeros_like(arr, dtype=np.uint8)
    for c, i in code_to_i.items():
        idx[arr == c] = i
    return idx


def panel(ax, name, cfg):
    minx, miny, maxx, maxy = cfg["bbox"]
    lon0, lat0, lon1, lat1 = transform_bounds("EPSG:32646", "EPSG:4326", minx, miny, maxx, maxy)
    dlon = (lon1 - lon0) * cfg["pad"]
    dlat = (lat1 - lat0) * cfg["pad"]
    win_lon0, win_lon1 = lon0 - dlon, lon1 + dlon
    win_lat0, win_lat1 = lat0 - dlat, lat1 + dlat
    with rasterio.open(cfg["tile"]) as src:
        win = from_bounds(win_lon0, win_lat0, win_lon1, win_lat1, transform=src.transform)
        arr = src.read(1, window=win, out_shape=(min(int(win.height), 1200), min(int(win.width), 1200)))
    idx = code_to_index(arr)
    extent = (win_lon0, win_lon1, win_lat0, win_lat1)
    ax.imshow(idx, cmap=cmap, vmin=0, vmax=len(codes_sorted) - 1, extent=extent, origin="upper", interpolation="nearest")
    rect = Rectangle((lon0, lat0), lon1 - lon0, lat1 - lat0, fill=False, edgecolor="black", linewidth=1.8, zorder=5)
    ax.add_patch(rect)
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.tick_params(labelsize=8)


handles = [plt.Rectangle((0, 0), 1, 1, color=WC_COLORS[c][1]) for c in codes_sorted if c in (10, 20, 30, 40, 50, 60, 80, 90)]
labels = [WC_COLORS[c][0] for c in codes_sorted if c in (10, 20, 30, 40, 50, 60, 80, 90)]

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
panel(axes[0], "Demra", SITES["Demra"])
panel(axes[1], "Uttar Khan", SITES["Uttar Khan"])
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=[0, 0.05, 1, 0.96])
fig.savefig(FIG_DIR / "figure1_study_area_demra_uttarkhan.png", dpi=1000, bbox_inches="tight")
plt.close(fig)
print("Saved figure1_study_area_demra_uttarkhan.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
panel(axes[0], "Hakaluki Haor", SITES["Hakaluki Haor"])
panel(axes[1], "Kaptai Lake", SITES["Kaptai Lake"])
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=[0, 0.05, 1, 0.96])
fig.savefig(FIG_DIR / "figureS1_locator_hakaluki_kaptai.png", dpi=1000, bbox_inches="tight")
plt.close(fig)
print("Saved figureS1_locator_hakaluki_kaptai.png")
