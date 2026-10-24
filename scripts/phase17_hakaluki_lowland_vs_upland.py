"""
Phase 17 -- Within-scene control for the Hakaluki flooded-forest signature.

Run over every available date (the recession time series). Phase 16 found Hakaluki tree cover at HH ~ -3.3 dB with HH-HV ~ 8 dB on both 23 Jun and
17 Jul 2026 (Fenchuganj stage at or above danger level on both). To show this is
double-bounce from flooded forest and not simply how trees look on this track, compare tree
cover in the haor basin (low ground) with tree cover on the surrounding hills/tilas (high
ground, cannot be inundated), in the SAME scene and a similar incidence-angle band.
Terrain from Copernicus GLO-30 (a DSM: includes canopy, so bands are approximate).

Outputs: processed/reanalysis_v2/phase17_hakaluki_lowland_upland.csv
"""
from pathlib import Path
import csv
import numpy as np
import h5py
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

ROOT = Path(r"D:\NISAR_Dhaka_Flood")
DL = Path(r"C:\Users\user\Downloads")
WC_TILE = ROOT / "raw_data" / "worldcover" / "ESA_WorldCover_10m_2020_v100_N24E090_Map.tif"
DEM = ("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/"
       "Copernicus_DSM_COG_10_N24_00_E092_00_DEM/Copernicus_DSM_COG_10_N24_00_E092_00_DEM.tif")
OUT = ROOT / "processed" / "reanalysis_v2" / "phase17_hakaluki_lowland_upland.csv"
CRS = "EPSG:32646"
GRID = "science/LSAR/GCOV/grids/frequencyA/"
# Hakaluki bbox widened by ~12 km east/west to take in the Patharia/Kulaura hill ranges
BOX = (387000.0, 2715000.0, 424000.0, 2740000.0)
SEARCH_DIRS = [ROOT / "raw_data" / "nisar_gcov", DL]


def dates():
    """All readable track-141/frame-014 DHDH granules, first readable copy per date."""
    import re
    out = {}
    for d in SEARCH_DIRS:
        for p in sorted(d.glob("NISAR_L2_PR_GCOV_*_141_A_014_4005_DHDH_*_001.h5")):
            date = re.search(r"_(2026\d{4})T", p.name).group(1)
            if date in out:
                continue
            try:
                h5py.File(p, "r").close()
                out[date] = p
            except OSError:
                pass
    return dict(sorted(out.items()))
BANDS = [("basin  < 12 m", -50, 12), ("12-25 m", 12, 25), ("hills  > 40 m", 40, 1000)]


def read(path):
    x0, y0, x1, y1 = BOX
    with h5py.File(path, "r") as f:
        x = f[GRID + "xCoordinates"][:]
        y = f[GRID + "yCoordinates"][:]
        xi = np.where((x >= x0) & (x <= x1))[0]
        yi = np.where((y >= y0) & (y <= y1))[0]
        sl = (slice(yi.min(), yi.max() + 1), slice(xi.min(), xi.max() + 1))
        hh, hv, m = f[GRID + "HHHH"][sl], f[GRID + "HVHV"][sl], f[GRID + "mask"][sl]
        dx, dy = x[1] - x[0], y[1] - y[0]
        t = from_origin(x[xi.min()] - dx / 2, y[yi.min()] - dy / 2, dx, -dy)
    return hh, hv, (m > 0) & (m < 255) & (hh > 0) & (hv > 0), t


def warp(src, shape, t, resampling):
    dst = np.full(shape, np.nan, dtype="float32")
    with rasterio.open(src) as s:
        reproject(rasterio.band(s, 1), dst, dst_transform=t, dst_crs=CRS, resampling=resampling,
                  src_nodata=s.nodata, dst_nodata=np.nan)
    return dst


rows = []
for date, path in dates().items():
    hh, hv, valid, t = read(path)
    wc = warp(WC_TILE, hh.shape, t, Resampling.nearest)
    dem = warp(DEM, hh.shape, t, Resampling.bilinear)
    tree = valid & (wc == 10) & np.isfinite(dem)
    print(f"\n{date}  (box {BOX}, tree px {tree.sum():,})")
    for label, lo, hi in BANDS:
        sel = tree & (dem >= lo) & (dem < hi)
        if sel.sum() < 200:
            print(f"  {label}: too few pixels ({sel.sum()})")
            continue
        h = 10 * np.log10(hh[sel].mean())
        v = 10 * np.log10(hv[sel].mean())
        frac = float((10 * np.log10(hh[sel]) > -5.9).mean() * 100)
        rows.append(dict(date=date, band=label.strip(), n=int(sel.sum()), hh_db=h, hv_db=v,
                         hh_minus_hv_db=h - v, pct_hh_gt_minus5p9=frac))
        print(f"  tree cover {label}: HH {h:6.2f}  HV {v:6.2f}  HH-HV {h - v:5.2f}  "
              f"HH>-5.9dB {frac:5.1f}%  n={sel.sum():,}")

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nWrote {OUT}")
