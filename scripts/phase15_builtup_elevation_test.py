"""
Phase 15 -- Is the co-pol brightening of built-up pixels on the peak-rain date a
waterlogging signal or a general wet-surface/rain effect?

Test: across Dhaka city (not only the two wards), stratify built-up pixels by terrain
height (Copernicus DEM GLO-30, read remotely as a COG) and compare the Exp1 change
(pre 30 Jun/1 Jul -> peak-rain 12/13 Jul) by elevation band. Street waterlogging collects
in low ground, so a flood signal should be strongest in the lowest bands; a uniform shift
across bands points to wet surfaces or a scene-level offset instead.

Uses the GCOV valid-sample mask (NISAR) and the HyP3 layover/shadow map (S1).
Outputs: processed/reanalysis_v2/phase15_builtup_by_elevation.csv
"""
from pathlib import Path
import csv
import numpy as np
import h5py
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_origin

ROOT = Path(r"D:\NISAR_Dhaka_Flood")
GCOV = ROOT / "raw_data" / "nisar_gcov"
S1 = ROOT / "processed" / "s1_rtc"
WC_TILE = ROOT / "raw_data" / "worldcover" / "ESA_WorldCover_10m_2020_v100_N21E090_Map.tif"
DEM = ("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/"
       "Copernicus_DSM_COG_10_N23_00_E090_00_DEM/Copernicus_DSM_COG_10_N23_00_E090_00_DEM.tif")
OUT = ROOT / "processed" / "reanalysis_v2" / "phase15_builtup_by_elevation.csv"

# Dhaka city core, EPSG:32646 (~90.35-90.48E, 23.70-23.88N)
CITY = (229000.0, 2622000.0, 243000.0, 2642000.0)
CRS = "EPSG:32646"
RES = 20.0
SHAPE = (int((CITY[3] - CITY[1]) / RES), int((CITY[2] - CITY[0]) / RES))
DST_T = from_origin(CITY[0], CITY[3], RES, RES)
GRID = "science/LSAR/GCOV/grids/frequencyA/"


def warp(src_path, band=1, resampling=Resampling.nearest, dtype="float32"):
    dst = np.full(SHAPE, np.nan if dtype == "float32" else 0, dtype=dtype)
    with rasterio.open(src_path) as src:
        reproject(rasterio.band(src, band), dst, dst_transform=DST_T, dst_crs=CRS,
                  resampling=resampling, src_nodata=src.nodata,
                  dst_nodata=np.nan if dtype == "float32" else 0)
    return dst


def nisar(pattern, term):
    path = [p for p in GCOV.glob(pattern) if p.suffix == ".h5"][0]
    with h5py.File(path, "r") as f:
        x = f[GRID + "xCoordinates"][:]
        y = f[GRID + "yCoordinates"][:]
        xi = np.where((x >= CITY[0]) & (x <= CITY[2]))[0]
        yi = np.where((y >= CITY[1]) & (y <= CITY[3]))[0]
        sl = (slice(yi.min(), yi.max() + 1), slice(xi.min(), xi.max() + 1))
        a = f[GRID + term][sl].astype("float32")
        m = f[GRID + "mask"][sl]
        dx, dy = x[1] - x[0], y[1] - y[0]
        t = from_origin(x[xi.min()] - dx / 2, y[yi.min()] - dy / 2, dx, -dy)
    a[(m == 0) | (m == 255) | ~(a > 0)] = np.nan
    # average linear power onto the common 20 m grid
    dst = np.full(SHAPE, np.nan, dtype="float32")
    reproject(a, dst, src_transform=t, src_crs=CRS, dst_transform=DST_T, dst_crs=CRS,
              resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan)
    return 10 * np.log10(dst)


def s1(stamp, pol):
    p = list(S1.glob(f"*{stamp}*/*_{pol}.tif"))[0]
    a = warp(p, resampling=Resampling.average)
    a[~(a > 0)] = np.nan
    return 10 * np.log10(a)


wc = warp(WC_TILE, dtype="uint8")
dem = warp(DEM, resampling=Resampling.bilinear)
ls = warp(list(S1.glob("*20260713T120342*/*_ls_map.tif"))[0], dtype="uint8")
built = wc == 50
bands = [(-10, 4), (4, 6), (6, 8), (8, 10), (10, 14), (14, 100)]

deltas = {
    "NISAR HH 30Jun->12Jul": nisar("*20260712T*", "HHHH") - nisar("*20260630T*", "HHHH"),
    "NISAR HV 30Jun->12Jul": nisar("*20260712T*", "HVHV") - nisar("*20260630T*", "HVHV"),
    "S1 VV 01Jul->13Jul": np.where(ls == 1, s1("20260713T120342", "VV") - s1("20260701T120341", "VV"), np.nan),
    "S1 VH 01Jul->13Jul": np.where(ls == 1, s1("20260713T120342", "VH") - s1("20260701T120341", "VH"), np.nan),
}

rows = []
print(f"grid {SHAPE}, built-up px {built.sum():,}, DEM range {np.nanmin(dem):.1f}-{np.nanmax(dem):.1f} m")
for name, d in deltas.items():
    line = []
    for lo, hi in bands:
        sel = built & np.isfinite(d) & (dem >= lo) & (dem < hi)
        n = int(sel.sum())
        mean = float(d[sel].mean()) if n >= 100 else np.nan
        frac = float((d[sel] > 3).mean() * 100) if n >= 100 else np.nan
        rows.append(dict(delta=name, dem_lo=lo, dem_hi=hi, n=n, mean_db=mean, pct_gt_plus3=frac))
        line.append(f"{lo}-{hi} m: {mean:+.2f} ({frac:.1f}% >+3, n={n})")
    # rank correlation across all built-up pixels
    sel = built & np.isfinite(d) & np.isfinite(dem)
    r = np.corrcoef(np.argsort(np.argsort(dem[sel])), np.argsort(np.argsort(d[sel])))[0, 1]
    print(f"\n{name}  (Spearman r with elevation = {r:+.3f}, n={sel.sum():,})\n  " + "\n  ".join(line))

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nWrote {OUT}")
