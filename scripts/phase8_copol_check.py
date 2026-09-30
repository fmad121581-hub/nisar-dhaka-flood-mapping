"""
Phase 8 -- Co-polarization (HH/VV) cross-check requested in response to external review.

Reviewer's point: the manuscript's flood-detection methodology uses cross-pol only
(HVHV for NISAR, VH for Sentinel-1). Classic double-bounce literature (Hess, Melack,
Filoso & Wang 1995, already cited in literature-review.md) shows the double-bounce
flooded-vegetation enhancement primarily in CO-pol (HH), with cross-pol used mainly to
separate woody/non-woody vegetation. This check asks: does the same delta pattern
(during - pre) show up in HH (NISAR) / VV (Sentinel-1) over the Uttar Khan positive-case
site, using files already on disk -- zero new downloads?

Scope: Experiment 1 (ascending track) ONLY. Experiment 2 (descending) is already excluded
from the manuscript's headline comparison because of the DHDH-vs-QPDH acquisition-mode
calibration artifact (see phase6-uttarkhan-positive-case-attempt.md) -- running a co-pol
check on an already-invalidated pair adds nothing.

Reuses the exact bbox, coregistration grid, and WorldCover classification from
phase6_uttarkhan.py; only the polarization channel changes (HVHV->HHHH for NISAR,
VH->VV for Sentinel-1).
"""

import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_origin
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds as _tb

_HOME = Path(os.environ["HOME"])
D_ROOT = _HOME / "mnt" / "d" / "NISAR_Dhaka_Flood"
PROC_DIR = D_ROOT / "processed"
OUT_DIR = PROC_DIR / "uttarkhan_case_copol"

UTTARKHAN_BBOX = (236582.74858659075, 2639492.080730549, 242585.99816525163, 2645681.772454119)  # EPSG:32646
TARGET_CRS = "EPSG:32646"
TARGET_RES = 20

WORLDCOVER_RAW = D_ROOT / "raw_data" / "worldcover" / "ESA_WorldCover_10m_2020_v100_N21E090_Map.tif"

WC_CLASSES = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland",
              50: "built-up", 60: "bare/sparse", 80: "permanent water", 90: "wetland"}
VEGETATED = [10, 20, 30, 40, 90]

NISAR_GCOV = D_ROOT / "raw_data" / "nisar_gcov"
S1_RTC = PROC_DIR / "s1_rtc"

# Experiment 1 (ascending) only -- co-pol channel
SOURCES = {
    "nisar_june30_asc": NISAR_GCOV / "NISAR_L2_PR_GCOV_024_069_A_014_4005_DHDH_A_20260630T232134_20260630T232209_P05023_N_F_J_001.h5",
    "nisar_july12_asc": NISAR_GCOV / "NISAR_L2_PR_GCOV_025_069_A_014_4005_DHDH_A_20260712T232134_20260712T232208_P05023_N_F_J_001.h5",
}
S1_SOURCES = {
    "s1_july1_asc": S1_RTC / "S1D_IW_20260701T120341_DVP_RTC20_G_gpuned_C6CB" / "S1D_IW_20260701T120341_DVP_RTC20_G_gpuned_C6CB_VV.tif",
    "s1_july13_asc": S1_RTC / "S1D_IW_20260713T120342_DVP_RTC20_G_gpuned_F738" / "S1D_IW_20260713T120342_DVP_RTC20_G_gpuned_F738_VV.tif",
}

NISAR_HH_DATASET = "science/LSAR/GCOV/grids/frequencyA/HHHH"

EXPERIMENT = {"nisar_pre": "nisar_june30_asc", "nisar_during": "nisar_july12_asc",
              "s1_pre": "s1_july1_asc", "s1_during": "s1_july13_asc"}


def extract_nisar(h5_path, out_tif, window_buffer_m=500.0):
    import h5py
    from rasterio.crs import CRS

    if out_tif.exists():
        print(f"  NISAR -> {out_tif.name}  (skip, already exists)")
        return
    minx, miny, maxx, maxy = UTTARKHAN_BBOX
    minx, miny, maxx, maxy = minx - window_buffer_m, miny - window_buffer_m, maxx + window_buffer_m, maxy + window_buffer_m
    with h5py.File(h5_path, "r") as f:
        g = f["science/LSAR/GCOV/grids/frequencyA"]
        dx = float(g["xCoordinateSpacing"][()]); dy = float(g["yCoordinateSpacing"][()])
        x0 = float(g["xCoordinates"][0]); y0 = float(g["yCoordinates"][0])
        epsg = int(g["projection"][()])
        n_rows, n_cols = f[NISAR_HH_DATASET].shape
        col0 = max(int((minx - x0) / dx), 0); col1 = min(int((maxx - x0) / dx) + 1, n_cols)
        row0 = max(int((maxy - y0) / dy), 0); row1 = min(int((miny - y0) / dy) + 1, n_rows)
        if col0 >= col1 or row0 >= row1:
            raise ValueError(f"Uttarkhan bbox doesn't overlap {h5_path.name}")
        power = f[NISAR_HH_DATASET][row0:row1, col0:col1]
        win_x0 = x0 + col0 * dx; win_y0 = y0 + row0 * dy
    transform = from_origin(win_x0 - dx / 2, win_y0 - dy / 2, dx, abs(dy))
    power = np.where(power > 0, power, np.nan)
    db = 10 * np.log10(power)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "height": db.shape[0], "width": db.shape[1], "count": 1,
               "dtype": "float32", "crs": CRS.from_epsg(epsg), "transform": transform}
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(db.astype("float32"), 1)
    print(f"  NISAR -> {out_tif.name}  ({db.shape[1]}x{db.shape[0]}px, {dx}m)")


def extract_s1(vv_tif, out_tif, window_buffer_m=500.0):
    if out_tif.exists():
        print(f"  S1 -> {out_tif.name}  (skip, already exists)")
        return
    minx, miny, maxx, maxy = UTTARKHAN_BBOX
    minx, miny, maxx, maxy = minx - window_buffer_m, miny - window_buffer_m, maxx + window_buffer_m, maxy + window_buffer_m
    with rasterio.open(vv_tif) as src:
        profile = src.profile.copy()
        if str(src.crs) != TARGET_CRS:
            minx, miny, maxx, maxy = _tb(TARGET_CRS, src.crs, minx, miny, maxx, maxy)
        win = from_bounds(minx, miny, maxx, maxy, transform=src.transform).round_offsets().round_lengths()
        power = src.read(1, window=win).astype("float64")
        win_transform = src.window_transform(win)
        win_crs = src.crs
    power = np.where(power > 0, power, np.nan)
    db = 10 * np.log10(power)
    profile.update(dtype="float32", count=1, height=db.shape[0], width=db.shape[1], transform=win_transform, crs=win_crs)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(db.astype("float32"), 1)
    print(f"  S1 -> {out_tif.name}  ({db.shape[1]}x{db.shape[0]}px, crs={win_crs})")


def coregister(src_tif, out_tif, resampling=Resampling.bilinear):
    minx, miny, maxx, maxy = UTTARKHAN_BBOX
    width = int(round((maxx - minx) / TARGET_RES)); height = int(round((maxy - miny) / TARGET_RES))
    if out_tif.exists():
        print(f"  (skip, already exists) {out_tif.name}")
        return width, height
    transform = from_origin(minx, maxy, TARGET_RES, TARGET_RES)
    with rasterio.open(src_tif) as src:
        profile = src.profile.copy()
        profile.update(crs=TARGET_CRS, transform=transform, width=width, height=height)
        out_tif.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_tif, "w", **profile) as dst:
            reproject(source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                      src_transform=src.transform, src_crs=src.crs, dst_transform=transform,
                      dst_crs=TARGET_CRS, resampling=resampling)
    return width, height


def read(p):
    with rasterio.open(p) as src:
        return src.read(1)


def class_stats(vals):
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return None
    return {"n": int(vals.size), "mean": float(np.mean(vals)), "median": float(np.median(vals)),
            "std": float(np.std(vals))}


def main():
    print("=== Step 1: extract HH (NISAR) / VV (S1) over Uttar Khan bbox, Experiment 1 only ===")
    raw_dir = OUT_DIR / "raw_extract"
    coreg_dir = OUT_DIR / "coregistered"
    for label, path in SOURCES.items():
        extract_nisar(path, raw_dir / f"{label}_HH.tif")
    for label, path in S1_SOURCES.items():
        extract_s1(path, raw_dir / f"{label}_VV.tif")

    print("\n=== Step 2: co-register to shared 20m grid (reuse same WorldCover coreg as HV run if present) ===")
    labels = [f"{l}_HH" for l in SOURCES] + [f"{l}_VV" for l in S1_SOURCES]
    for label in labels:
        w, h = coregister(raw_dir / f"{label}.tif", coreg_dir / f"{label}_coreg.tif")
    wc_coreg = PROC_DIR / "uttarkhan_case" / "coregistered" / "worldcover_coreg.tif"
    if not wc_coreg.exists():
        coregister(WORLDCOVER_RAW, coreg_dir / "worldcover_coreg.tif", resampling=Resampling.nearest)
        wc_coreg = coreg_dir / "worldcover_coreg.tif"
    print(f"  Grid: {w}x{h}px @ {TARGET_RES}m, worldcover from {wc_coreg}")

    classes = read(wc_coreg).astype(int)

    print("\n=== Step 3: co-pol delta (during - pre), Experiment 1, vegetated classes ===")
    for sensor, suffix in (("nisar", "HH"), ("s1", "VV")):
        pre = read(coreg_dir / f"{EXPERIMENT[f'{sensor}_pre']}_{suffix}_coreg.tif")
        during = read(coreg_dir / f"{EXPERIMENT[f'{sensor}_during']}_{suffix}_coreg.tif")
        delta = during - pre
        print(f"\n  {sensor.upper()} {suffix} delta ({EXPERIMENT[f'{sensor}_during']} - {EXPERIMENT[f'{sensor}_pre']}):")
        for code in VEGETATED:
            name = WC_CLASSES[code]
            mask = classes == code
            vals = delta[mask]
            s = class_stats(vals)
            if s:
                v2 = vals[~np.isnan(vals)]
                frac_up = float(np.mean(v2 > 3.0)) * 100
                frac_down = float(np.mean(v2 < -3.0)) * 100
                print(f"    {name:16s} n={s['n']:6d}  mean={s['mean']:+.2f}dB  median={s['median']:+.2f}dB  "
                      f"std={s['std']:.2f}  >+3dB:{frac_up:5.1f}%  <-3dB:{frac_down:5.1f}%")


if __name__ == "__main__":
    main()
