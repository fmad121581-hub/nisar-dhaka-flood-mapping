"""
Phase 10 -- Hakaluki Haor (Sprint B positive-control site): first differencing run.

Site: Hakaluki Haor wetland, Sylhet division (~24.58-24.73N, 92.00-92.13E),
Bangladesh's largest freshwater marsh, chosen as a positive-control site
because extreme monsoon inundation there is physically near-certain even
though hyperlocal news corroboration was NOT found for the July 2026 event
(see hakaluki-haor-corroboration-check.md -- treat any "flooded" framing as
inference from the extreme rainfall record, not as confirmed by an
independent source).

Rainfall context (flood-event-timeline-check.md / sprint-ab-plan.md, NASA
POWER daily precipitation at the site centroid): the "pre" and "during"
scenes bracket a jump from a wet monsoon baseline to an extreme multi-day
deluge, not a dry-to-flooded contrast. Interpret any signal accordingly.

Sources:
  NISAR L2 GCOV, track/frame 141/014, DHDH_A (dual-pol, ascending):
    pre    = 2026-06-23  (NISAR_..._023_141_A_014_...)
    during = 2026-07-17  (NISAR_..._025_141_A_014_...)
  Sentinel-1 IW, ascending (matching orbit direction), via ASF HyP3 RTC_GAMMA,
  20m pixel spacing, gamma0 power scale:
    pre    = 2026-06-26  (S1D_IW_20260626T115549..._C752)
    during = 2026-07-20  (S1D_IW_20260720T115551..._B073)
  Both S1 products are read directly out of their delivered zip archives via
  GDAL's /vsizip/ virtual filesystem -- no full extraction -- since extracting
  the ~500MB VH GeoTIFF from each ~1GB zip was too slow for this session's
  per-command time budget. rasterio opens /vsizip/<zip>/<member> exactly like
  a normal path and only reads the bytes covering the requested window.

Cross-pol only (NISAR HVHV / S1 VH), matching the manuscript's primary
methodology (co-pol VV cross-check is a follow-up, not required for the
Sprint B go/no-go read).

WorldCover tile N24E090 (already on D:) covers this site -- confirmed in
phase7-newsite-template.md notes, distinct from the N21E090 tile used for
Demra/Uttar Khan.
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
OUT_DIR = PROC_DIR / "hakaluki_haor_case"

# EPSG:32646 (UTM 46N), reprojected from the WGS84 estimate in sprint-ab-plan.md
# (24.5833-24.7333N, 92.00-92.1333E) via rasterio.warp.transform_bounds.
HAKALUKI_BBOX = (398749.23462869856, 2719084.5651329057, 412351.523585242, 2735786.232745403)
TARGET_CRS = "EPSG:32646"
TARGET_RES = 20

WORLDCOVER_RAW = D_ROOT / "raw_data" / "worldcover" / "ESA_WorldCover_10m_2020_v100_N24E090_Map.tif"

WC_CLASSES = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland",
              50: "built-up", 60: "bare/sparse", 80: "permanent water", 90: "wetland"}
VEGETATED = [10, 20, 30, 40, 90]

NISAR_HV_DATASET = "science/LSAR/GCOV/grids/frequencyA/HVHV"

NISAR_SOURCES = {
    "pre": DOWNLOADS / "NISAR_L2_PR_GCOV_023_141_A_014_4005_DHDH_A_20260623T231316_20260623T231350_P05023_N_F_J_001.h5",
    "during": DOWNLOADS / "NISAR_L2_PR_GCOV_025_141_A_014_4005_DHDH_A_20260717T231314_20260717T231349_P05023_N_F_J_001.h5",
}

def _vsizip(zip_name, member):
    return f"/vsizip/{DOWNLOADS / zip_name}/{Path(zip_name).stem}/{member}"

S1_SOURCES = {
    "pre": _vsizip("S1D_IW_20260626T115549_DVP_RTC20_G_gpuned_C752.zip",
                    "S1D_IW_20260626T115549_DVP_RTC20_G_gpuned_C752_VH.tif"),
    "during": _vsizip("S1D_IW_20260720T115551_DVP_RTC20_G_gpuned_B073.zip",
                       "S1D_IW_20260720T115551_DVP_RTC20_G_gpuned_B073_VH.tif"),
}


def extract_nisar(h5_path, out_tif, window_buffer_m=500.0):
    import h5py
    from rasterio.crs import CRS

    if out_tif.exists():
        print(f"  NISAR -> {out_tif.name}  (skip, already exists)")
        return
    minx, miny, maxx, maxy = HAKALUKI_BBOX
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
            raise ValueError(f"Hakaluki bbox doesn't overlap {h5_path.name} -- wrong track/frame")
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
    print(f"  NISAR -> {out_tif.name}  ({db.shape[1]}x{db.shape[0]}px, {dx}m, epsg={epsg})")


def extract_s1(vh_path, out_tif, window_buffer_m=500.0):
    if out_tif.exists():
        print(f"  S1 -> {out_tif.name}  (skip, already exists)")
        return
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds as _tb
    minx, miny, maxx, maxy = HAKALUKI_BBOX
    minx, miny, maxx, maxy = minx - window_buffer_m, miny - window_buffer_m, maxx + window_buffer_m, maxy + window_buffer_m
    with rasterio.open(vh_path) as src:
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
    minx, miny, maxx, maxy = HAKALUKI_BBOX
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


def water_calibration_check(coreg_dir, sensor):
    classes = read(coreg_dir / "worldcover_coreg.tif").astype(int)
    water_mask = classes == 80
    pre = read(coreg_dir / f"{sensor}_pre_coreg.tif")
    during = read(coreg_dir / f"{sensor}_during_coreg.tif")
    delta = during - pre
    vals = delta[water_mask]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        print(f"  {sensor}: WARNING -- no permanent-water pixels in this bbox, calibration check not possible")
        return None
    shift = float(np.mean(vals))
    print(f"  {sensor}: permanent-water shift (during - pre) = {shift:+.2f} dB (n={vals.size})")
    return shift


def main():
    raw_dir = OUT_DIR / "raw_extract"
    coreg_dir = OUT_DIR / "coregistered"

    print("=== Step 1: extract rasters (windowed reads, no full unzip) ===")
    for label, path in NISAR_SOURCES.items():
        extract_nisar(path, raw_dir / f"nisar_{label}.tif")
    for label, path in S1_SOURCES.items():
        extract_s1(path, raw_dir / f"s1_{label}.tif")

    print("\n=== Step 2: co-register to shared 20m grid ===")
    w = h = None
    for label in NISAR_SOURCES:
        w, h = coregister(raw_dir / f"nisar_{label}.tif", coreg_dir / f"nisar_{label}_coreg.tif")
    for label in S1_SOURCES:
        w, h = coregister(raw_dir / f"s1_{label}.tif", coreg_dir / f"s1_{label}_coreg.tif")
    coregister(WORLDCOVER_RAW, coreg_dir / "worldcover_coreg.tif", resampling=Resampling.nearest)
    print(f"  Grid: {w}x{h}px @ {TARGET_RES}m")

    classes = read(coreg_dir / "worldcover_coreg.tif").astype(int)
    total_px = classes.size
    print("\n  WorldCover class mix over Hakaluki Haor bbox:")
    for code in sorted(WC_CLASSES):
        frac = float(np.mean(classes == code)) * 100
        if frac > 0.01:
            print(f"    {WC_CLASSES[code]:16s} {frac:5.1f}%")

    print("\n=== Step 3: permanent-water calibration check ===")
    water_calibration_check(coreg_dir, "nisar")
    water_calibration_check(coreg_dir, "s1")

    print("\n=== Step 4: delta (during - pre), cross-pol, per vegetated class ===")
    for sensor in ("nisar", "s1"):
        pre = read(coreg_dir / f"{sensor}_pre_coreg.tif")
        during = read(coreg_dir / f"{sensor}_during_coreg.tif")
        delta = during - pre
        label = "NISAR HVHV" if sensor == "nisar" else "S1 VH"
        print(f"\n  {label} delta (during - pre):")
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
