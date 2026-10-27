"""
Figure 2 (v2): the two granules rejected by screening.
(a) 14 Jul 2026 QPDH partial frame (091/077) over Uttar Khan: GCOV valid-sample mask.
(b) Same window, HV backscatter as delivered (invalid pixels included).
(c) Class-mean HH at Hakaluki on every track-141 date, showing the 22 Aug outlier, with each
    granule's HHHH `mean_value` attribute annotated.
"""
from pathlib import Path
import csv
import datetime as dt
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap

ROOT = Path(r"D:\NISAR_Dhaka_Flood")
GCOV = ROOT / "raw_data" / "nisar_gcov"
OUT = ROOT / "figures" / "figure_v2_02_granule_screening.png"
UK = (236582.74858659075, 2639492.080730549, 242585.99816525163, 2645681.772454119)
G = "science/LSAR/GCOV/grids/frequencyA/"

p14 = next(GCOV.glob("*20260714T*_001.h5"))
with h5py.File(p14, "r") as f:
    x, y = f[G + "xCoordinates"][:], f[G + "yCoordinates"][:]
    pad = 3000
    xi = np.where((x >= UK[0] - pad) & (x <= UK[2] + pad))[0]
    yi = np.where((y >= UK[1] - pad) & (y <= UK[3] + pad))[0]
    sl = (slice(yi.min(), yi.max() + 1), slice(xi.min(), xi.max() + 1))
    mask, hv = f[G + "mask"][sl], f[G + "HVHV"][sl]
ext = (x[xi[0]] / 1e3, x[xi[-1]] / 1e3, y[yi[-1]] / 1e3, y[yi[0]] / 1e3)

fig = plt.figure(figsize=(11, 7.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
ax = fig.add_subplot(gs[0, 0])
cat = np.where(mask == 0, 0, np.where(mask == 255, 2, 1))
# colour-blind safe pair: orange = invalid (0), dark blue = valid
ax.imshow(cat, extent=ext, cmap=ListedColormap(["#e66101", "#2166ac", "#cccccc"]), vmin=0, vmax=2,
          interpolation="nearest")
ax.add_patch(plt.Rectangle((UK[0] / 1e3, UK[1] / 1e3), (UK[2] - UK[0]) / 1e3, (UK[3] - UK[1]) / 1e3,
                           fill=False, ec="black", lw=1.5))
ax.text(0.02, 0.98, "(a) mask, 14 Jul (091/077, partial frame)\norange = 0 (invalid), blue = valid",
        transform=ax.transAxes, va="top", fontsize=8.5, bbox=dict(fc="white", ec="none", alpha=0.85))
ax.set_xlabel("Easting (km)")
ax.set_ylabel("Northing (km)")

ax = fig.add_subplot(gs[0, 1])
with np.errstate(divide="ignore", invalid="ignore"):
    im = ax.imshow(10 * np.log10(hv), extent=ext, cmap="gray", vmin=-30, vmax=-5, interpolation="nearest")
ax.add_patch(plt.Rectangle((UK[0] / 1e3, UK[1] / 1e3), (UK[2] - UK[0]) / 1e3, (UK[3] - UK[1]) / 1e3,
                           fill=False, ec="#e66101", lw=1.5))
ax.text(0.02, 0.98, "(b) HV as delivered (dB)", transform=ax.transAxes, va="top", fontsize=8.5,
        bbox=dict(fc="white", ec="none", alpha=0.85))
ax.set_xlabel("Easting (km)")
fig.colorbar(im, ax=ax, shrink=0.8, label="HV (dB)")

ax = fig.add_subplot(gs[1, :])
rows = [r for r in csv.DictReader(open(ROOT / "processed" / "reanalysis_v2" / "phase16_hakaluki_timeseries.csv"))]
attrs = {}
for p in list(GCOV.glob("*_141_A_014_*_001.h5")) + list(Path(r"C:\Users\user\Downloads").glob("*_141_A_014_*_001.h5")):
    d = p.name.split("_")[11][:8]
    if d in attrs:
        continue
    try:
        with h5py.File(p, "r") as f:
            attrs[d] = float(f[G + "HHHH"].attrs["mean_value"])
    except OSError:
        pass
for cls, col in (("cropland", "#b5651d"), ("permanent water", "#1d4e89"), ("tree cover", "#1b4332")):
    sub = sorted((r["date"], float(r["hh_db"])) for r in rows if r["cls"] == cls)
    ax.plot([dt.datetime.strptime(d, "%Y%m%d") for d, _ in sub], [v for _, v in sub], "o-", color=col, label=cls)
for d, v in sorted(attrs.items()):
    ax.annotate(f"{v:.3g}", (dt.datetime.strptime(d, "%Y%m%d"), 26 if d == "20260822" else -0.5),
                ha="center", fontsize=7.5, color="#d62728" if d == "20260822" else "#555")
ax.text(0.01, 0.97, "(c) Hakaluki class-mean HH by date; numbers = granule HHHH mean_value attribute",
        transform=ax.transAxes, va="top", fontsize=8.5)
ax.set_ylabel("class-mean HH (dB)")
ax.set_ylim(-20, 30)
ax.legend(fontsize=8, loc="center left")
ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"Wrote {OUT}; attrs {attrs}")
