"""
Phase 20 -- Sentinel-1 (OPERA RTC-S1, track 41 ascending) at Kaptai Lake, matched to the
NISAR reservoir-rise analysis of phase 19 (26 Jun, 8 Jul, 20 Jul, 1 Aug 2026).

Bursts 086047-086050 (IW1/IW2) are mosaicked per date onto one 30 m grid over the Kaptai box
(first valid burst wins), so overlap is counted once. OPERA mask 0 = valid.

Outputs: processed/reanalysis_v2/phase20_kaptai_s1.csv
"""
from pathlib import Path
import csv
import re
from collections import defaultdict
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

ROOT = Path(r"D:\NISAR_Dhaka_Flood")
OPERA = ROOT / "raw_data" / "opera_rtc_s1"
WC_TILE = ROOT / "raw_data" / "worldcover" / "ESA_WorldCover_10m_2020_v100_N21E090_Map.tif"
OUT = ROOT / "processed" / "reanalysis_v2" / "phase20_kaptai_s1.csv"
BOX = (402869.90916713024, 2485087.8977870205, 433476.6124612646, 2549171.3867336917)
CRS = "EPSG:32646"
RES = 30.0
SHAPE = (int((BOX[3] - BOX[1]) / RES), int((BOX[2] - BOX[0]) / RES))
T = from_origin(BOX[0], BOX[3], RES, RES)
WC = {10: "tree cover", 30: "grassland", 40: "cropland", 90: "wetland", 80: "permanent water"}


def to_grid(path, resampling):
    dst = np.full(SHAPE, np.nan, dtype="float32")
    with rasterio.open(path) as s:
        reproject(rasterio.band(s, 1), dst, dst_transform=T, dst_crs=CRS, resampling=resampling,
                  src_nodata=s.nodata, dst_nodata=np.nan)
    return dst


wc = np.zeros(SHAPE, np.uint8)
with rasterio.open(WC_TILE) as s:
    reproject(rasterio.band(s, 1), wc, dst_transform=T, dst_crs=CRS, resampling=Resampling.mode)

by_date = defaultdict(list)
KAPTAI_BURSTS = ("086047", "086048", "086049", "086050")
for vv in sorted(OPERA.glob("OPERA_L2_RTC-S1_T041-*_VV.tif")):
    if not any(f"T041-{b}-" in vv.name for b in KAPTAI_BURSTS):
        continue
    date = re.search(r"_(2026\d{4})T", vv.name).group(1)
    by_date[date].append(vv)

# NISAR frame-013 valid footprint (30 Jun) on the same grid, for a like-for-like comparison
import h5py
from rasterio.transform import from_origin as _fo
NISAR_30JUN = next((ROOT / "raw_data" / "nisar_gcov").glob("NISAR_L2_PR_GCOV_024_069_A_013_*_001.h5"))
with h5py.File(NISAR_30JUN, "r") as f:
    g = "science/LSAR/GCOV/grids/frequencyA/"
    x, y = f[g + "xCoordinates"][:], f[g + "yCoordinates"][:]
    xi = np.where((x >= BOX[0]) & (x <= BOX[2]))[0]
    yi = np.where((y >= BOX[1]) & (y <= BOX[3]))[0]
    m = f[g + "mask"][yi.min():yi.max() + 1, xi.min():xi.max() + 1]
    nt = _fo(x[xi.min()] - (x[1] - x[0]) / 2, y[yi.min()] - (y[1] - y[0]) / 2, x[1] - x[0], y[0] - y[1])
nis_valid = np.zeros(SHAPE, np.uint8)
reproject(((m > 0) & (m < 255)).astype(np.uint8), nis_valid, src_transform=nt, src_crs=CRS,
          dst_transform=T, dst_crs=CRS, resampling=Resampling.mode)
nis_valid = nis_valid.astype(bool)
print(f"NISAR frame-013 valid footprint: {nis_valid.mean():.1%} of the box")

rows = []
for date, paths in sorted(by_date.items()):
    VV = np.full(SHAPE, np.nan, "float32")
    VH = np.full(SHAPE, np.nan, "float32")
    for vv in paths:
        vh = Path(str(vv).replace("_VV.tif", "_VH.tif"))
        mk = Path(str(vv).replace("_VV.tif", "_mask.tif"))
        a, b = to_grid(vv, Resampling.bilinear), to_grid(vh, Resampling.bilinear)
        m = to_grid(mk, Resampling.nearest)
        ok = (m == 0) & (a > 0) & (b > 0) & np.isnan(VV)
        VV[ok], VH[ok] = a[ok], b[ok]
    cover = np.isfinite(VV).mean()
    print(f"\nS1 {date}  bursts={len(paths)}  box covered {cover:.1%}")
    for scope, fp in (("full box", np.ones(SHAPE, bool)), ("NISAR footprint", nis_valid)):
      print(f"  [{scope}]")
      for code, name in WC.items():
        sel = (wc == code) & np.isfinite(VV) & np.isfinite(VH) & fp
        if sel.sum() < 100:
            continue
        c, x = 10 * np.log10(VV[sel].mean()), 10 * np.log10(VH[sel].mean())
        dark = float((10 * np.log10(VV[sel]) < -15).mean() * 100)
        rows.append(dict(sensor="S1", scope=scope, date=date, cls=name, n=int(sel.sum()), co_db=c, cross_db=x,
                         co_minus_cross_db=c - x, pct_co_below_minus15=dark))
        print(f"    {name:16s} VV {c:6.2f}  VH {x:6.2f}  VV<-15dB {dark:5.1f}%  n={sel.sum():,}")

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nWrote {OUT}")
