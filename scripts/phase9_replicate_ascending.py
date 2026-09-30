"""
Phase 9 -- Replicate the ascending-track (mode-consistent) comparison at Uttar Khan
using a second, independent post-baseline acquisition, extending Sprint A's one
reliable pair (n=1) to n=2 pairs sharing a common pre-flood baseline.

Background: exp1_peak_ascending in phase6_uttarkhan.py compared
nisar_june30_asc (pre) -> nisar_july12_asc (during, ~peak rain per the
NASA POWER record) and s1_july1_asc (pre) -> s1_july13_asc (during), all
DHDH_A / same ascending track+frame -- the only pair in the whole dataset
free of the QPDH/DHDH acquisition-mode mismatch that invalidated
exp2_recession_descending.

Per the archive inventory (nisar-archive-inventory-demra-uttarkhan.md),
NISAR cycle 026 (2026-07-24, DHDH_A, same track/frame) and the matching
Sentinel-1 ascending scene (2026-07-25) extend that same mode-consistent
series with a THIRD acquisition date, sharing the identical pre-flood
baseline (june30_asc / july1_asc). This script adds that second
during/post comparison: june30_asc/july1_asc (pre) -> july24_asc/july25_asc.

Per flood-event-timeline-check.md section 6, the NASA POWER rainfall
record shows the heavy July 7-13 deluge (309mm/week) dropping off sharply
from July 14, with a smaller second rain pulse around July 20-23. July 24
therefore is NOT a clean dry post-flood state -- it follows that second,
smaller pulse by ~1-4 days. This is a recession/second-pulse comparison
against the same pre-flood baseline, not a clean flood-vs-no-flood
contrast; treat accordingly when interpreting results, exactly as the
July 12/13 "during" scenes required reframing from "flooded" to
"peak-rain/early-recession".

Reuses the exact bbox, coregistration grid, WorldCover classification,
and pre-flood baseline rasters already produced by phase6_uttarkhan.py
(same OUT_DIR, so the existing coregistered pre/baseline/worldcover
files are picked up via the same skip-if-exists cache rather than
recomputed) -- only the new during-scene rasters are extracted here.
Cross-pol (HVHV/VH) only, matching the manuscript's primary methodology.
"""

import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

_HOME = Path(os.environ["HOME"])
D_ROOT = _HOME / "mnt" / "d" / "NISAR_Dhaka_Flood"
DOWNLOADS = _HOME / "mnt" / "Downloads"
PROC_DIR = D_ROOT / "processed"
OUT_DIR = PROC_DIR / "uttarkhan_case"  # SAME as phase6 -- reuse cached pre/baseline + worldcover

UTTARKHAN_BBOX = (236582.74858659075, 2639492.080730549, 242585.99816525163, 2645681.772454119)  # EPSG:32646
TARGET_CRS = "EPSG:32646"
TARGET_RES = 20

WORLDCOVER_RAW = D_ROOT / "raw_data" / "worldcover" / "ESA_WorldCover_10m_2020_v100_N21E090_Map.tif"

WC_CLASSES = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland",
              50: "built-up", 60: "bare/sparse", 80: "permanent water", 90: "wetland"}
VEGETATED = [10, 20, 30, 40, 90]

# Pre-flood baseline -- identical files/labels to phase6_uttarkhan.py so the
# existing cached raw_extract/coregistered outputs are reused, not recomputed.
BASELINE_SOURCES = {
    "nisar_june30_asc": DOWNLOADS / "NISAR_L2_PR_GCOV_024_069_A_014_4005_DHDH_A_20260630T232134_20260630T232209_P05023_N_F_J_001.h5",
}
BASELINE_S1_SOURCES = {
    "s1_july1_asc": Path("/tmp/s1_extract/S1D_IW_20260701T120341_DVP_RTC20_G_gpuned_C6CB/S1D_IW_20260701T120341_DVP_RTC20_G_gpuned_C6CB_VH.tif"),
}

# New second during/post acquisition -- cycle 026, same track/frame/mode (DHDH_A)
NEW_SOURCES = {
    "nisar_july24_asc": D_ROOT / "raw_data" / "nisar_gcov" / "NISAR_L2_PR_GCOV_026_069_A_014_4005_DHDH_A_20260724T232133_20260724T232208_P05023_N_F_J_001.h5",
}
NEW_S1_SOURCES = {
    "s1_july25_asc": PROC_DIR / "s1_rtc" / "S1D_IW_20260725T120343_DVP_RTC20_G_gpuned_1E6A" / "S1D_IW_20260725T120343_DVP_RTC20_G_gpuned_1E6A_VH.tif",
}

NISAR_HV_DATASET = "science/LSAR/GCOV/grids/frequencyA/HVHV"

EXPERIMENT = {"nisar_pre": "nisar_june30_asc", "nisar_during": "nisar_july24_asc",
              "s1_pre": "s1_july1_asc", "s1_during": "s1_july25_asc"}


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
        n_rows, n_cols = f[NISAR_HV_DATASET].shape
        col0 = max(int((minx - x0) / dx), 0); col1 = min(int((maxx - x0) / dx) + 1, n_cols)
        row0 = max(int((maxy - y0) / dy), 0); row1 = min(int((miny - y0) / dy) + 1, n_rows)
        if col0 >= col1 or row0 >= row1:
            raise ValueError(f"Uttarkhan bbox doesn't overlap {h5_path.name}")
        power = f[NISAR_HV_DATASET][row0:row1, col0:col1]
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


def extract_s1(vh_tif, out_tif, window_buffer_m=500.0):
    if out_tif.exists():
        print(f"  S1 -> {out_tif.name}  (skip, already exists)")
        return
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds as _tb
    minx, miny, maxx, maxy = UTTARKHAN_BBOX
    minx, miny, maxx, maxy = minx - window_buffer_m, miny - window_buffer_m, maxx + window_buffer_m, maxy + window_buffer_m
    with rasterio.open(vh_tif) as src:
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
    print("=== Step 1: extract baseline (reuse cache) + new during-scene rasters ===")
    raw_dir = OUT_DIR / "raw_extract"
    coreg_dir = OUT_DIR / "coregistered"
    for label, path in {**BASELINE_SOURCES, **NEW_SOURCES}.items():
        extract_nisar(path, raw_dir / f"{label}.tif")
    for label, path in {**BASELINE_S1_SOURCES, **NEW_S1_SOURCES}.items():
        extract_s1(path, raw_dir / f"{label}.tif")

    print("\n=== Step 2: co-register (reuse cache for baseline/worldcover) ===")
    all_labels = list(BASELINE_SOURCES) + list(NEW_SOURCES) + list(BASELINE_S1_SOURCES) + list(NEW_S1_SOURCES)
    for label in all_labels:
        w, h = coregister(raw_dir / f"{label}.tif", coreg_dir / f"{label}_coreg.tif")
    coregister(WORLDCOVER_RAW, coreg_dir / "worldcover_coreg.tif", resampling=Resampling.nearest)
    print(f"  Grid: {w}x{h}px @ {TARGET_RES}m")

    classes = read(coreg_dir / "worldcover_coreg.tif").astype(int)

    print("\n=== Step 3: delta (during - pre), cross-pol, replicate-ascending pair ===")
    print(f"  Pair: {EXPERIMENT['nisar_during']} - {EXPERIMENT['nisar_pre']} (NISAR HVHV) / "
          f"{EXPERIMENT['s1_during']} - {EXPERIMENT['s1_pre']} (S1 VH)")
    for sensor in ("nisar", "s1"):
        pre = read(coreg_dir / f"{EXPERIMENT[f'{sensor}_pre']}_coreg.tif")
        during = read(coreg_dir / f"{EXPERIMENT[f'{sensor}_during']}_coreg.tif")
        delta = during - pre
        print(f"\n  {sensor.upper()} delta ({EXPERIMENT[f'{sensor}_during']} - {EXPERIMENT[f'{sensor}_pre']}):")
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
