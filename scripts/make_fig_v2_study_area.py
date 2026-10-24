"""
Figure 1 (v2): study sites. (a) Bangladesh with district boundaries and the three settings;
(b) Hakaluki Haor, (c) Kaptai Lake with the NISAR frame-013 valid footprint, (d) Dhaka wards,
each on ESA WorldCover 2020. No title is drawn inside the image (NHESS asks for captions only).
Boundaries: geoBoundaries BGD ADM0/ADM2 (CC BY 4.0).
"""
from pathlib import Path
import numpy as np
import geopandas as gpd
import h5py
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle, Patch

ROOT = Path(r"D:\NISAR_Dhaka_Flood")
B = ROOT / "raw_data" / "boundaries"
WC21 = ROOT / "raw_data" / "worldcover" / "ESA_WorldCover_10m_2020_v100_N21E090_Map.tif"
WC24 = ROOT / "raw_data" / "worldcover" / "ESA_WorldCover_10m_2020_v100_N24E090_Map.tif"
OUT = ROOT / "figures" / "figure_v2_01_study_area.png"
CRS = "EPSG:32646"

BOXES = {
    "Demra": (240673.90374, 2621657.94409, 246404.89660, 2630128.18446),
    "Uttar Khan": (236582.74858659075, 2639492.080730549, 242585.99816525163, 2645681.772454119),
    "Hakaluki Haor": (398749.23462869856, 2719084.5651329057, 412351.523585242, 2735786.232745403),
    "Kaptai Lake": (402869.90916713024, 2485087.8977870205, 433476.6124612646, 2549171.3867336917),
}
FENCHUGANJ = (92.017, 24.703)  # upazila town on the Kushiyara; approximate gauge location

WC_CODES = [10, 20, 30, 40, 50, 60, 80, 90]
WC_NAMES = ["tree cover", "shrubland", "grassland", "cropland", "built-up", "bare/sparse",
            "permanent water", "herbaceous wetland"]
WC_COLS = ["#006400", "#ffbb22", "#ffff4c", "#f096ff", "#fa0000", "#b4b4b4", "#0064c8", "#0096a0"]
cmap = ListedColormap(["#ffffff"] + WC_COLS)
norm = BoundaryNorm([0, 5, 15, 25, 35, 45, 55, 70, 85, 95], cmap.N)


def ll(box):
    return transform_bounds(CRS, "EPSG:4326", *box)


def wc_panel(ax, tile, box, step=30.0):
    x0, y0, x1, y1 = box
    shape = (int((y1 - y0) / step), int((x1 - x0) / step))
    t = from_origin(x0, y1, step, step)
    a = np.zeros(shape, np.uint8)
    with rasterio.open(tile) as s:
        reproject(rasterio.band(s, 1), a, dst_transform=t, dst_crs=CRS, resampling=Resampling.mode)
    ax.imshow(a, cmap=cmap, norm=norm, extent=(x0 / 1e3, x1 / 1e3, y0 / 1e3, y1 / 1e3),
              interpolation="nearest")
    ax.set_xlabel("Easting (km, UTM 46N)")
    ax.set_ylabel("Northing (km)")
    ax.tick_params(labelsize=7)


adm0 = gpd.read_file(B / "geoBoundaries-BGD-ADM0_simplified.geojson")
adm2 = gpd.read_file(B / "geoBoundaries-BGD-ADM2_simplified.geojson")

fig = plt.figure(figsize=(11, 8.2))
gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1])
ax_a = fig.add_subplot(gs[:, 0])
adm2.boundary.plot(ax=ax_a, lw=0.3, color="#999")
adm0.boundary.plot(ax=ax_a, lw=0.8, color="#333")
for name, box in BOXES.items():
    lo0, la0, lo1, la1 = ll(box)
    ax_a.add_patch(Rectangle((lo0, la0), lo1 - lo0, la1 - la0, fill=False, ec="#c0392b", lw=1.4))
ax_a.annotate("Dhaka wards", ll(BOXES["Uttar Khan"])[:2], xytext=(88.6, 24.2), fontsize=8,
              arrowprops=dict(arrowstyle="-", lw=0.6))
ax_a.annotate("Hakaluki Haor", ll(BOXES["Hakaluki Haor"])[2:], xytext=(91.0, 25.6), fontsize=8,
              arrowprops=dict(arrowstyle="-", lw=0.6))
ax_a.annotate("Kaptai Lake", ll(BOXES["Kaptai Lake"])[:2], xytext=(90.6, 21.6), fontsize=8,
              arrowprops=dict(arrowstyle="-", lw=0.6))
ax_a.plot(*FENCHUGANJ, "^", color="#1d4e89", ms=6)
ax_a.annotate("Fenchuganj gauge", FENCHUGANJ, xytext=(90.9, 25.2), fontsize=8, color="#1d4e89",
              arrowprops=dict(arrowstyle="-", lw=0.6, color="#1d4e89"))
ax_a.set_xlim(88.0, 92.8)
ax_a.set_ylim(20.5, 26.7)
ax_a.set_aspect(1 / np.cos(np.deg2rad(23.7)))
ax_a.set_xlabel("Longitude (°E)")
ax_a.set_ylabel("Latitude (°N)")
ax_a.text(0.03, 0.97, "(a)", transform=ax_a.transAxes, va="top", fontweight="bold")

ax_b = fig.add_subplot(gs[0, 1])
wc_panel(ax_b, WC24, BOXES["Hakaluki Haor"])
ax_b.text(0.03, 0.97, "(b) Hakaluki Haor", transform=ax_b.transAxes, va="top", fontsize=9,
          fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.8))

ax_c = fig.add_subplot(gs[:, 2])
wc_panel(ax_c, WC21, BOXES["Kaptai Lake"], step=40.0)
nis = next((ROOT / "raw_data" / "nisar_gcov").glob("NISAR_L2_PR_GCOV_024_069_A_013_*_001.h5"))
x0, y0, x1, y1 = BOXES["Kaptai Lake"]
with h5py.File(nis, "r") as f:
    g = "science/LSAR/GCOV/grids/frequencyA/"
    x, y = f[g + "xCoordinates"][:], f[g + "yCoordinates"][:]
    xi = np.where((x >= x0) & (x <= x1))[0][::8]
    yi = np.where((y >= y0) & (y <= y1))[0][::8]
    m = f[g + "mask"][yi.min():yi.max() + 1:8, xi.min():xi.max() + 1:8]
valid = ((m > 0) & (m < 255)).astype(float)
ax_c.contourf(x[xi] / 1e3, y[yi] / 1e3, valid, levels=[-0.5, 0.5], colors="none", hatches=["////"])
ax_c.contourf(x[xi] / 1e3, y[yi] / 1e3, valid, levels=[-0.5, 0.5], colors="white", alpha=0.45)
ax_c.contour(x[xi] / 1e3, y[yi] / 1e3, valid, levels=[0.5], colors="black", linewidths=1.2)
ax_c.text(0.03, 0.99, "(c) Kaptai Lake\nhatched: outside NISAR 069/013", transform=ax_c.transAxes,
          va="top", fontsize=9, fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.8))

ax_d = fig.add_subplot(gs[1, 1])
dx0 = min(BOXES["Demra"][0], BOXES["Uttar Khan"][0]) - 3000
dy0 = BOXES["Demra"][1] - 3000
dx1 = max(BOXES["Demra"][2], BOXES["Uttar Khan"][2]) + 3000
dy1 = BOXES["Uttar Khan"][3] + 3000
wc_panel(ax_d, WC21, (dx0, dy0, dx1, dy1), step=20.0)
for name in ("Demra", "Uttar Khan"):
    a0, b0, a1, b1 = BOXES[name]
    ax_d.add_patch(Rectangle((a0 / 1e3, b0 / 1e3), (a1 - a0) / 1e3, (b1 - b0) / 1e3, fill=False,
                             ec="black", lw=1.4))
    ax_d.text(a0 / 1e3, b1 / 1e3 + 0.3, name, fontsize=8, fontweight="bold")
ax_d.text(0.03, 0.03, "(d) Dhaka", transform=ax_d.transAxes, fontsize=9, fontweight="bold",
          bbox=dict(fc="white", ec="none", alpha=0.8))

fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(WC_COLS, WC_NAMES)], loc="lower center",
           ncol=8, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT, dpi=200)
print(f"Wrote {OUT}")
