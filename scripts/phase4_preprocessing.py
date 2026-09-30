"""
Phase 4 -- Analysis-ready preprocessing for the redesigned dual-epoch, dual-track experiments
NISAR L-band vs Sentinel-1 C-band flood detection -- Demra ward, Dhaka

Per phase4-redesign-dry-baseline.md:
  Experiment 1 (peak flood, ascending):     NISAR Jun30 (dry) -> NISAR Jul12 (already processed)
                                             S1    Jul1  (dry) -> S1    Jul13 (already processed)
  Experiment 2 (early recession, descending): NISAR Jul2  (dry) -> NISAR Jul14 (already processed)
                                             S1    Jul3  (dry) -> S1    Jul15 (already processed)

This script only needs to produce the FOUR NEW "dry baseline" rasters:
  nisar_june30_asc_HVHV.tif, nisar_july2_desc_HVHV.tif, s1_july1_asc_VH.tif, s1_july3_desc_VH.tif
following the exact same stages/parameters as phase2_preprocessing.py so they are
directly comparable to the four rasters already sitting in processed/analysis_ready/.

Reuses stage2/3/4/5 logic verbatim from phase2_preprocessing.py (same dataset key,
same target grid, same masking classes) -- only the input granules differ.
"""

import os
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.crs import CRS

# ---------------------------------------------------------------------------
# CONFIG -- identical target grid/CRS/masking as Phase 2, for direct comparability
#
# NOTE: this script runs through the device-bridge Linux shell (not native Windows
# Python), so Windows-style paths (C:\..., D:\...) are wrong here -- pathlib on a
# POSIX system treats backslashes as literal characters, not separators. Use the
# bridge's own mount points instead ($HOME/mnt/Downloads, $HOME/mnt/d/...), which
# map to the same files as C:\Users\user\Downloads and D:\ on the actual PC.
# ---------------------------------------------------------------------------
_HOME = Path(os.environ["HOME"])
DOWNLOADS_DIR = _HOME / "mnt" / "Downloads"
PROC_DIR = _HOME / "mnt" / "d" / "NISAR_Dhaka_Flood" / "processed"
S1_RTC_DIR = PROC_DIR / "s1_rtc"
WORLDCOVER_CLIPPED = PROC_DIR / "worldcover_clipped.tif"

DEMRA_BBOX = (240673.90374, 2621657.94409, 246404.89660, 2630128.18446)  # EPSG:32646, meters
TARGET_CRS = "EPSG:32646"
TARGET_RESOLUTION = 20
WORLDCOVER_BUILTUP_CLASS = 50
WORLDCOVER_WATER_CLASS = 80

NISAR_HV_DATASET = "science/LSAR/GCOV/grids/frequencyA/HVHV"

# New dry-baseline NISAR granules (still sitting in Downloads, not yet moved to raw_data/nisar_gcov
# due to bridge copy speed -- referencing Downloads directly is fine, path is only read from here)
NISAR_NEW = {
    "june30_asc": DOWNLOADS_DIR
    / "NISAR_L2_PR_GCOV_024_069_A_014_4005_DHDH_A_20260630T232134_20260630T232209_P05023_N_F_J_001.h5",
    "july2_desc": DOWNLOADS_DIR
    / "NISAR_L2_PR_GCOV_024_091_D_077_2005_DHDH_M_20260702T123457_20260702T123531_P05023_N_F_J_001.h5",
}

# S1 HyP3 RTC job info (submitted 2026-09-26 14:41 UTC via direct HyP3 API call from the browser session)
S1_NEW_JOBS = {
    "july1_asc": {
        "job_id": "9f0595a8-2a0a-4b4e-b9a2-a52f5d97f642",
        "granule": "S1D_IW_SLC__1SDV_20260701T120341_20260701T120408_003480_006271_8772",
    },
    "july3_desc": {
        "job_id": "77645bfb-c129-4c6e-a7d3-ed9f47e03b68",
        "granule": "S1D_IW_SLC__1SDV_20260703T235521_20260703T235547_003516_0063B1_AE3C",
    },
}


# ---------------------------------------------------------------------------
# STAGE 2 (NISAR) -- identical to phase2_preprocessing.stage2_extract_nisar_band
# ---------------------------------------------------------------------------
def stage2_extract_nisar_band(h5_path: Path, out_tif: Path, dataset_key: str = NISAR_HV_DATASET,
                               window_bbox=DEMRA_BBOX, window_buffer_m: float = 500.0):
    import h5py

    grid_group = str(Path(dataset_key).parent)
    with h5py.File(h5_path, "r") as f:
        g = f[grid_group]
        dx = float(g["xCoordinateSpacing"][()])
        dy = float(g["yCoordinateSpacing"][()])
        x0 = float(g["xCoordinates"][0])
        y0 = float(g["yCoordinates"][0])
        epsg = int(g["projection"][()])
        n_rows, n_cols = f[dataset_key].shape

        if window_bbox is not None:
            minx, miny, maxx, maxy = window_bbox
            minx, miny = minx - window_buffer_m, miny - window_buffer_m
            maxx, maxy = maxx + window_buffer_m, maxy + window_buffer_m
            col0 = int((minx - x0) / dx)
            col1 = int((maxx - x0) / dx) + 1
            row0 = int((maxy - y0) / dy)
            row1 = int((miny - y0) / dy) + 1
            col0, row0 = max(col0, 0), max(row0, 0)
            col1, row1 = min(col1, n_cols), min(row1, n_rows)
            if col0 >= col1 or row0 >= row1:
                raise ValueError(f"window_bbox {window_bbox} doesn't overlap {h5_path.name}'s grid")
            power = f[dataset_key][row0:row1, col0:col1]
            win_x0 = x0 + col0 * dx
            win_y0 = y0 + row0 * dy
        else:
            power = f[dataset_key][:]
            win_x0, win_y0 = x0, y0

    transform = from_origin(win_x0 - dx / 2, win_y0 - dy / 2, dx, abs(dy))
    crs = CRS.from_epsg(epsg)

    power = np.where(power > 0, power, np.nan)
    db = 10 * np.log10(power)

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "height": db.shape[0], "width": db.shape[1], "count": 1,
               "dtype": "float32", "crs": crs, "transform": transform}
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(db.astype("float32"), 1)
    print(f"Wrote {out_tif}  (EPSG:{epsg}, {dx}m pixels)")


# ---------------------------------------------------------------------------
# STAGE 3 (S1) -- windowed read added for Phase 4 (the new dry-baseline RTC
# outputs are full-swath, ~14200x11200px -- reading + float64 conversion of the
# whole array OOM-killed this box's 3.8GB bridge shell. Phase 2's originals were
# presumably processed on a machine with more RAM; this window-crop keeps the
# result numerically identical over Demra (same crop stage4 would apply anyway)
# while fitting in memory here.
# ---------------------------------------------------------------------------
def stage3_extract_s1_band(rtc_vh_tif: Path, out_tif: Path, window_bbox=DEMRA_BBOX, window_buffer_m: float = 500.0):
    from rasterio.windows import from_bounds

    with rasterio.open(rtc_vh_tif) as src:
        profile = src.profile.copy()
        if window_bbox is not None and src.crs is not None and str(src.crs) == TARGET_CRS:
            minx, miny, maxx, maxy = window_bbox
            minx, miny = minx - window_buffer_m, miny - window_buffer_m
            maxx, maxy = maxx + window_buffer_m, maxy + window_buffer_m
            win = from_bounds(minx, miny, maxx, maxy, transform=src.transform).round_offsets().round_lengths()
            power = src.read(1, window=win).astype("float64")
            win_transform = src.window_transform(win)
        else:
            power = src.read(1).astype("float64")
            win_transform = src.transform

    power = np.where(power > 0, power, np.nan)
    db = 10 * np.log10(power)

    profile.update(dtype="float32", count=1, height=db.shape[0], width=db.shape[1], transform=win_transform)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(db.astype("float32"), 1)
    print(f"Wrote {out_tif}  ({db.shape[1]}x{db.shape[0]} px, windowed to Demra bbox+buffer)")


# ---------------------------------------------------------------------------
# STAGE 4 -- identical co-registration to shared Demra grid
# ---------------------------------------------------------------------------
def stage4_coregister(src_tif: Path, out_tif: Path, ref_bounds=DEMRA_BBOX, ref_crs=TARGET_CRS,
                       ref_res=TARGET_RESOLUTION):
    minx, miny, maxx, maxy = ref_bounds
    width = int(round((maxx - minx) / ref_res))
    height = int(round((maxy - miny) / ref_res))
    transform = from_origin(minx, maxy, ref_res, ref_res)

    with rasterio.open(src_tif) as src:
        profile = src.profile.copy()
        profile.update(crs=ref_crs, transform=transform, width=width, height=height)
        out_tif.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_tif, "w", **profile) as dst:
            reproject(source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                      src_transform=src.transform, src_crs=src.crs, dst_transform=transform,
                      dst_crs=ref_crs, resampling=Resampling.bilinear)
    print(f"Co-registered -> {out_tif}  ({width}x{height} px @ {ref_res}m, clipped to Demra bbox)")


# ---------------------------------------------------------------------------
# STAGE 5 -- identical masking using the already-clipped WorldCover raster
# ---------------------------------------------------------------------------
def stage5_mask_water_builtup(backscatter_tif: Path, worldcover_tif: Path, out_tif: Path):
    with rasterio.open(backscatter_tif) as src:
        data = src.read(1)
        profile = src.profile.copy()
    with rasterio.open(worldcover_tif) as wc:
        classes = wc.read(1)

    mask_out = (classes == WORLDCOVER_BUILTUP_CLASS) | (classes == WORLDCOVER_WATER_CLASS)
    data = np.where(mask_out, np.nan, data)

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)
    print(f"Masked -> {out_tif}")


def stage6_save(name: str, data_tif: Path):
    final_dir = PROC_DIR / "analysis_ready"
    final_dir.mkdir(parents=True, exist_ok=True)
    dest = final_dir / f"{name}.tif"
    import shutil
    shutil.copy(data_tif, dest)
    print(f"Analysis-ready: {dest}")


def run_nisar_pipeline(label: str, h5_path: Path):
    print(f"\n=== NISAR {label} ===")
    hv_tif = PROC_DIR / "nisar_hv" / f"{label}_HVHV.tif"
    coreg_tif = PROC_DIR / "coregistered" / f"nisar_{label}_HVHV_coreg.tif"
    masked_tif = PROC_DIR / "masked" / f"nisar_{label}_HVHV_masked.tif"
    stage2_extract_nisar_band(h5_path, hv_tif)
    stage4_coregister(hv_tif, coreg_tif)
    stage5_mask_water_builtup(coreg_tif, WORLDCOVER_CLIPPED, masked_tif)
    stage6_save(f"nisar_{label}_HVHV", masked_tif)


def run_s1_pipeline(label: str, rtc_vh_tif: Path):
    print(f"\n=== Sentinel-1 {label} ===")
    db_tif = PROC_DIR / "s1_db" / f"{label}_VH_db.tif"
    coreg_tif = PROC_DIR / "coregistered" / f"s1_{label}_VH_coreg.tif"
    masked_tif = PROC_DIR / "masked" / f"s1_{label}_VH_masked.tif"
    stage3_extract_s1_band(rtc_vh_tif, db_tif)
    stage4_coregister(db_tif, coreg_tif)
    stage5_mask_water_builtup(coreg_tif, WORLDCOVER_CLIPPED, masked_tif)
    stage6_save(f"s1_{label}_VH", masked_tif)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "nisar"
    if which == "nisar":
        for label, h5_path in NISAR_NEW.items():
            run_nisar_pipeline(label, h5_path)
    elif which == "s1":
        # arg2 must be the path to the downloaded RTC _VH.tif
        label = sys.argv[2]
        vh_tif = Path(sys.argv[3])
        run_s1_pipeline(label, vh_tif)
    else:
        print("usage: phase4_preprocessing.py [nisar|s1 <label> <vh_tif_path>]")
