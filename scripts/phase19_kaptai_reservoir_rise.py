"""
Phase 19 -- Kaptai Lake as an inundation case with a known water-level rise.

Lake level (BPDB via press): 73.9 ft on 10 Jun; 84.6 ft on 8 Jul; ~99-100.6 ft on 12-13 Jul;
102.3 ft on 14 Jul; 103.8 ft on 19 Jul; 103.3 ft on 22 Jul 2026 (spillway gates open 18-22 Jul).
NISAR track 069 frame 013 (DHDH, ascending): 30 Jun, 12 Jul, 24 Jul -- before, during and
after most of the ~9 m rise. Shoreline grassland/cropland/wetland should go dark (specular)
as it drowns; hill tree cover (73% of the box) is the stable reference.

Sentinel-1 counterpart: the HyP3 RTC VV/VH extracts from phase 12 (26 Jun -> 20 Jul), already
co-registered with WorldCover in processed/kaptai_lake_case/coregistered.

Outputs: processed/reanalysis_v2/phase19_kaptai.csv
"""
from pathlib import Path
import csv
import re
import numpy as np
import h5py
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

ROOT = Path(r"D:\NISAR_Dhaka_Flood")
GCOV = ROOT / "raw_data" / "nisar_gcov"
WC_TILE = ROOT / "raw_data" / "worldcover" / "ESA_WorldCover_10m_2020_v100_N21E090_Map.tif"
S1_DIR = ROOT / "processed" / "kaptai_lake_case" / "coregistered"
OUT = ROOT / "processed" / "reanalysis_v2" / "phase19_kaptai.csv"
KAPTAI_BBOX = (402869.90916713024, 2485087.8977870205, 433476.6124612646, 2549171.3867336917)
CRS = "EPSG:32646"
GRID = "science/LSAR/GCOV/grids/frequencyA/"
WC = {10: "tree cover", 30: "grassland", 40: "cropland", 90: "wetland", 80: "permanent water"}


def nisar(path):
    x0, y0, x1, y1 = KAPTAI_BBOX
    with h5py.File(path, "r") as f:
        x = f[GRID + "xCoordinates"][:]
        y = f[GRID + "yCoordinates"][:]
        xi = np.where((x >= x0) & (x <= x1))[0]
        yi = np.where((y >= y0) & (y <= y1))[0]
        sl = (slice(yi.min(), yi.max() + 1), slice(xi.min(), xi.max() + 1))
        hh, hv, m = f[GRID + "HHHH"][sl], f[GRID + "HVHV"][sl], f[GRID + "mask"][sl]
        qa = float(f[GRID + "HHHH"].attrs["mean_value"])
        full = f["science/LSAR/identification/isFullFrame"][()].decode()
        dx, dy = x[1] - x[0], y[1] - y[0]
        t = from_origin(x[xi.min()] - dx / 2, y[yi.min()] - dy / 2, dx, -dy)
    return hh, hv, (m > 0) & (m < 255) & (hh > 0) & (hv > 0), t, qa, full


def db(a):
    return 10 * np.log10(a.mean()) if a.size else np.nan


rows = []
wc = None
for p in sorted(GCOV.glob("NISAR_L2_PR_GCOV_*_069_A_013_*_001.h5")):
    date = re.search(r"_(2026\d{4})T", p.name).group(1)
    hh, hv, valid, t, qa, full = nisar(p)
    if wc is None:
        wc = np.zeros(hh.shape, np.uint8)
        with rasterio.open(WC_TILE) as s:
            reproject(rasterio.band(s, 1), wc, dst_transform=t, dst_crs=CRS, resampling=Resampling.nearest)
    print(f"\nNISAR {date}  full_frame={full}  HHHH mean_value attr={qa:.3g}  valid {valid.mean():.1%}")
    for code, name in WC.items():
        sel = valid & (wc == code)
        h, v = db(hh[sel]), db(hv[sel])
        dark = float((10 * np.log10(hh[sel]) < -15).mean() * 100)  # open-water-like share
        rows.append(dict(sensor="NISAR", date=date, cls=name, n=int(sel.sum()), co_db=h, cross_db=v,
                         co_minus_cross_db=h - v, pct_co_below_minus15=dark))
        print(f"  {name:16s} HH {h:6.2f}  HV {v:6.2f}  HH<-15dB {dark:5.1f}%  n={sel.sum():,}")

# Sentinel-1 (phase 12 co-registered rasters, already in dB? detect from value range)
wc_s1 = rasterio.open(S1_DIR / "worldcover_coreg.tif").read(1)
for role, date in (("pre", "20260626"), ("during", "20260720")):
    vv = rasterio.open(S1_DIR / f"s1vv_{role}_coreg.tif").read(1).astype("float64")
    vh = rasterio.open(S1_DIR / f"s1_{role}_coreg.tif").read(1).astype("float64")
    if np.nanmax(vv) < 5:          # stored in dB -> back to linear for class means
        vv, vh = 10 ** (vv / 10), 10 ** (vh / 10)
    print(f"\nS1 {date} ({role})")
    for code, name in WC.items():
        sel = (wc_s1 == code) & np.isfinite(vv) & np.isfinite(vh) & (vv > 0) & (vh > 0)
        c, x = db(vv[sel]), db(vh[sel])
        dark = float((10 * np.log10(vv[sel]) < -15).mean() * 100)
        rows.append(dict(sensor="S1", date=date, cls=name, n=int(sel.sum()), co_db=c, cross_db=x,
                         co_minus_cross_db=c - x, pct_co_below_minus15=dark))
        print(f"  {name:16s} VV {c:6.2f}  VH {x:6.2f}  VV<-15dB {dark:5.1f}%  n={sel.sum():,}")

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nWrote {OUT}")
