import os
from pathlib import Path
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HOME = Path(os.environ["HOME"])
D_ROOT = _HOME / "mnt" / "d" / "NISAR_Dhaka_Flood"
PROC = D_ROOT / "processed"
FIG_DIR = D_ROOT / "figures"

data = np.load(PROC / "qa_spatial_arrays.npz")

def get_extent(path):
    with rasterio.open(path) as src:
        b = src.bounds
        return (b.left, b.right, b.bottom, b.top)

panels = [
    ("A. Uttar Khan, mode-mismatched\n(2 Jul DHDH -> 14 Jul QPDH)", data["uk_mismatch_delta"], data["uk_mismatch_valid"],
     PROC/"uttarkhan_case"/"coregistered"/"nisar_july2_desc_coreg.tif"),
    ("B. Uttar Khan, mode-matched\n(30 Jun DHDH -> 12 Jul DHDH)", data["uk_match_delta"], data["uk_match_valid"],
     PROC/"uttarkhan_case"/"coregistered"/"nisar_june30_asc_coreg.tif"),
    ("C. Hakaluki Haor, Sentinel-1 VV\n(26 Jun -> 20 Jul)", data["hh_vv_delta"], data["hh_vv_valid"],
     PROC/"hakaluki_haor_case_copol"/"coregistered"/"s1_pre_coreg.tif"),
    ("D. Kaptai Lake, Sentinel-1 VV\n(26 Jun -> 20 Jul)", data["kl_vv_delta"], data["kl_vv_valid"],
     PROC/"kaptai_lake_case"/"coregistered"/"s1vv_pre_coreg.tif"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
vmax = 6
im = None
for ax, (title, delta, valid, ref_path) in zip(axes.flat, panels):
    extent = get_extent(ref_path)
    masked = np.where(valid, delta, np.nan)
    masked = np.clip(masked, -vmax, vmax)
    im = ax.imshow(masked, cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=extent, origin="upper")
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=10)
    ax.set_xlabel("Easting (m, EPSG:32646)", fontsize=8)
    ax.set_ylabel("Northing (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(style="plain", axis="both")
    x0, x1, y0, y1 = extent
    real_mean = float(np.nanmean(np.where(valid, delta, np.nan)))
    ax.text(0.98, 0.98, "mean = {:+.2f} dB\nn = {:,} px".format(real_mean, int(valid.sum())),
            transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"))
    bar_len = round((x1 - x0) * 0.2, -2) or 100
    bx0 = x0 + (x1 - x0) * 0.05
    by0 = y0 + (y1 - y0) * 0.05
    ax.plot([bx0, bx0 + bar_len], [by0, by0], color="black", linewidth=2.5, solid_capstyle="butt")
    ax.text(bx0 + bar_len/2, by0 + (y1-y0)*0.02, "{} m".format(int(bar_len)), ha="center", va="bottom", fontsize=7)

fig.subplots_adjust(hspace=0.35, wspace=0.3, top=0.90, right=0.86)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.03)
cbar.set_label("Permanent-water delta backscatter (dB), during minus pre (clipped at +/-6 dB)", fontsize=9)
fig.suptitle("Figure 8. Spatial pattern of the permanent-water calibration check, four scene pairs", fontsize=13, y=0.97)
fig.savefig(FIG_DIR / "figure8_qa_spatial_maps.png", dpi=1000, bbox_inches="tight")
plt.close(fig)
print("Saved figure8_qa_spatial_maps.png")
