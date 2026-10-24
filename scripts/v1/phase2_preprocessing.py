"""
Phase 2 -- Analysis-Ready Data preprocessing pipeline
NISAR L-band vs Sentinel-1 C-band flood detection -- Demra ward, Dhaka
Case study: mid-July 2026 Dhaka monsoon flood (~July 12-16, 2026)

Per the project workflow (Phase 2 -- Analysis-Ready Data):
  - Use RTC (radiometrically terrain-corrected) products where possible
  - Co-register NISAR and Sentinel-1 backscatter layers to a common projection/grid
  - Mask permanent water, buildings, and irrelevant areas
  - Avoid comparing unprocessed raw data from the two sensors directly

Run order:
  1. stage1_submit_rtc_jobs()        -- submit Sentinel-1 SLC to ASF HyP3 for RTC (one-time, ~15-60 min turnaround)
  2. stage1b_download_rtc_outputs()  -- once HyP3 jobs finish, pull the RTC GeoTIFFs down
  3. explore_h5_structure()          -- run once on a NISAR GCOV file to confirm the exact
                                         internal dataset paths (NISAR's HDF5 layout needs to be
                                         confirmed against the real file -- see note in that function)
  4. stage2_extract_nisar_band()     -- pull the HV cross-pol backscatter band out of GCOV, convert to dB
  5. stage3_extract_s1_band()        -- pull the VH cross-pol backscatter band out of the RTC GeoTIFF, convert to dB
  6. stage4_coregister()             -- reproject both to one shared grid over Demra ward
  7. stage5_mask_water_builtup()     -- mask out permanent water / built-up pixels using ESA WorldCover
  8. stage6_save()                   -- write the analysis-ready rasters Phase 3 will difference

Needs: pip install hyp3_sdk rasterio h5py geopandas numpy
"""

import getpass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

# ---------------------------------------------------------------------------
# CONFIG -- adjust paths if you move the raw_data folder
# ---------------------------------------------------------------------------
RAW_DIR = Path(r"D:\NISAR_Dhaka_Flood\raw_data")
NISAR_DIR = RAW_DIR / "nisar_gcov"
S1_DIR = RAW_DIR / "sentinel1_slc"

PROC_DIR = Path(r"D:\NISAR_Dhaka_Flood\processed")
S1_RTC_DIR = PROC_DIR / "s1_rtc"

NISAR_GRANULES = {
    # July 12 ascending, dual-pol (H, HV) -- just before the flood window
    "july12_asc_dualpol": NISAR_DIR
    / "NISAR_L2_PR_GCOV_025_069_A_014_4005_DHDH_A_20260712T232134_20260712T232208_P05023_N_F_J_001.h5",
    # July 14 descending, quad-pol -- inside the flood window
    "july14_desc_quadpol": NISAR_DIR
    / "NISAR_L2_PR_GCOV_025_091_D_077_2005_QPDH_A_20260714T123456_20260714T123512_P05023_F_P_J_001.h5",
}

# HyP3 submits by ASF granule name (no file extension), not by local path --
# it reprocesses from ASF's own archive copy, so the SLC zips we already
# downloaded are a reference copy, not a required input to this step.
S1_GRANULE_NAMES = {
    "july13_asc": "S1D_IW_SLC__1SDV_20260713T120342_20260713T120409_003655_006865_0BF6",
    "july15_desc": "S1D_IW_SLC__1SDV_20260715T235521_20260715T235548_003691_00699F_84B5",
}

# Already on this PC (used by informal-settlement-classifier / Flood_risk / Urban_Heat_island_efect too)
WORLDCOVER_ZIP = Path(
    r"C:\Users\user\Downloads\ESA_WorldCover_10m_2020_v100_60deg_macrotile_S30E060.zip"
)
WORLDCOVER_BUILTUP_CLASS = 50   # ESA WorldCover class code
WORLDCOVER_WATER_CLASS = 80     # ESA WorldCover class code

# Demra ward bounding box -- extracted from ward_labeled.gpkg (informal-settlement-classifier
# project), combining the 3 GADM sub-units where NAME_3='Demra' (fid 99 "Matuail (part)",
# fid 124 "Saralia", fid 165 "Demra (part)"). That file's own SRS (confirmed via its
# gpkg_spatial_ref_sys table) is EPSG:32646 -- the SAME UTM zone already used as TARGET_CRS
# below -- so these numbers are already meters in ref_crs, no lon/lat conversion needed.
# CAVEAT: NAME_3='Demra' is a GADM Thana/Upazila-level unit, not confirmed to line up with
# DSCC's official numbered ward system used elsewhere (e.g. "Ward 49"). Flood_risk's
# ward_risk.shp is a second candidate worth checking if DSCC ward numbering matters later.
DEMRA_BBOX = (240673.90374, 2621657.94409, 246404.89660, 2630128.18446)  # minx, miny, maxx, maxy (EPSG:32646, meters)
TARGET_CRS = "EPSG:32646"   # UTM zone 46N -- correct zone for Dhaka
TARGET_RESOLUTION = 20      # meters -- common ground sampling so both sensors are compared fairly


# ---------------------------------------------------------------------------
# STAGE 1 -- Sentinel-1 RTC via ASF HyP3 (on-demand cloud processing)
# ---------------------------------------------------------------------------
def stage1_submit_rtc_jobs():
    """Submit the two Sentinel-1 SLC scenes to ASF HyP3 for RTC processing.

    This is the brief's recommended "first processing tool" for getting Sentinel-1
    to RTC without hand-rolling SNAP/ISCE3. Jobs run in ASF's cloud; typical
    turnaround is 15-60 minutes. You'll need the same NASA Earthdata Login used
    for the downloads (username fahim121581).
    """
    import hyp3_sdk as sdk

    hyp3 = sdk.HyP3(
        username=input("NASA Earthdata username: "),
        password=getpass.getpass("NASA Earthdata password: "),
    )

    jobs = sdk.Batch()
    for label, granule_name in S1_GRANULE_NAMES.items():
        jobs += hyp3.submit_rtc_job(
            granule_name,
            name=f"nisar_dhaka_flood_{label}",
            resolution=TARGET_RESOLUTION,
            radiometry="gamma0",
            scale="power",
            speckle_filter=False,
            dem_matching=False,
            include_dem=False,
            include_inc_map=False,
        )
    print(f"Submitted {len(jobs)} HyP3 RTC jobs.")
    print("Poll status with: jobs = hyp3.watch(jobs)  (blocks until done)")
    print("...or check https://search.asf.alaska.edu -> 'My Files' in the browser.")
    return hyp3, jobs


def stage1b_download_rtc_outputs(hyp3, jobs, out_dir: Path = S1_RTC_DIR):
    """Once jobs report complete, pull the RTC GeoTIFFs down to out_dir.
    Each job produces (at least) a *_VV.tif and *_VH.tif -- VH is the cross-pol
    band we actually need for the double-bounce comparison.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = hyp3.watch(jobs)  # blocks until every job is SUCCEEDED or FAILED
    file_list = jobs.download_files(str(out_dir))
    print(f"Downloaded {len(file_list)} files to {out_dir}")
    return file_list


# ---------------------------------------------------------------------------
# STAGE 2 -- NISAR L2 GCOV: find and extract the HV cross-pol band
# ---------------------------------------------------------------------------
def explore_h5_structure(h5_path: Path):
    """Print the full dataset tree of a NISAR GCOV .h5 file. Mostly useful now as a
    sanity check -- the real layout has already been confirmed against both actual
    granules (see notes below), so stage2_extract_nisar_band() no longer needs this
    run first.
    """
    import h5py

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}  shape={obj.shape}  dtype={obj.dtype}")

    with h5py.File(h5_path, "r") as f:
        print(f"Structure of {h5_path.name}:")
        f.visititems(visitor)


# Confirmed against the real July12 GCOV file (dual-pol, frequencyA):
#   science/LSAR/GCOV/grids/frequencyA/HHHH, HVHV  -- shape (33552, 34200), float32
#   listOfPolarizations = [HH, HV]; listOfCovarianceTerms = [HHHH, HVHV]
#   projection = 32646 (EPSG) -- SAME UTM zone as TARGET_CRS, no reprojection needed here
#   xCoordinateSpacing = 10.0, yCoordinateSpacing = -10.0 (10m pixels, north-up)
#   x range ~59765-401755, y range ~2572565-2908075 -- Demra bbox falls well inside this
# The quad-pol July14 file adds VVVV/VHVH alongside HHHH/HVHV under the same grid group;
# HVHV is present in both, so it's used as the common cross-pol band for both dates.
NISAR_HV_DATASET = "science/LSAR/GCOV/grids/frequencyA/HVHV"


def stage2_extract_nisar_band(
    h5_path: Path,
    out_tif: Path,
    dataset_key: str = NISAR_HV_DATASET,
    window_bbox=DEMRA_BBOX,
    window_buffer_m: float = 500.0,
):
    """Extract the HV cross-pol covariance band from a GCOV file, convert linear
    power to dB, and write it out as a proper georeferenced GeoTIFF.

    A full NISAR GCOV frequencyA band covers a whole ~340km orbit swath at 10m --
    that's a ~34000x33500 float32 array, ~4.3GB, which doesn't fit in memory on a
    lot of machines (it didn't fit here). Since Phase 2 only needs the Demra ward
    window anyway (stage4_coregister crops to it later), read that window straight
    off disk via h5py's slicing instead of loading the full array -- h5py only
    touches the requested hyperslab, it doesn't materialize the whole dataset.
    window_buffer_m pads the crop so stage4's resampling/co-registration has a
    little edge margin to work with; pass window_bbox=None to force a full read.

    dataset_key defaults to the confirmed HVHV path; pass a different one only if
    a future granule's layout differs (re-check with explore_h5_structure()).
    """
    import h5py
    from rasterio.transform import from_origin
    from rasterio.crs import CRS

    grid_group = str(Path(dataset_key).parent)  # .../grids/frequencyA

    with h5py.File(h5_path, "r") as f:
        g = f[grid_group]
        dx = float(g["xCoordinateSpacing"][()])
        dy = float(g["yCoordinateSpacing"][()])  # negative (north-up)
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
            row0 = int((maxy - y0) / dy)  # dy negative, larger y -> smaller row
            row1 = int((miny - y0) / dy) + 1
            col0, row0 = max(col0, 0), max(row0, 0)
            col1, row1 = min(col1, n_cols), min(row1, n_rows)
            if col0 >= col1 or row0 >= row1:
                raise ValueError(f"window_bbox {window_bbox} doesn't overlap {h5_path.name}'s grid")
            power = f[dataset_key][row0:row1, col0:col1]
            # top-left corner of the *window*, not the full scene
            win_x0 = x0 + col0 * dx
            win_y0 = y0 + row0 * dy
        else:
            power = f[dataset_key][:]
            win_x0, win_y0 = x0, y0

    # xCoordinates/yCoordinates (and the window origin above) are pixel-center
    # coords; shift by half a pixel to get the raster's top-left corner.
    transform = from_origin(win_x0 - dx / 2, win_y0 - dy / 2, dx, abs(dy))
    crs = CRS.from_epsg(epsg)

    power = np.where(power > 0, power, np.nan)
    db = 10 * np.log10(power)

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": db.shape[0],
        "width": db.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
    }
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(db.astype("float32"), 1)
    print(f"Wrote {out_tif}  (EPSG:{epsg}, {dx}m pixels)")


# ---------------------------------------------------------------------------
# STAGE 3 -- Sentinel-1 RTC GeoTIFF: extract VH, convert to dB
# ---------------------------------------------------------------------------
def stage3_extract_s1_band(rtc_vh_tif: Path, out_tif: Path):
    """HyP3 RTC output is already power-scale gamma0. Convert to dB and re-save
    (kept as a separate step so both sensors go through an identical dB conversion)."""
    with rasterio.open(rtc_vh_tif) as src:
        power = src.read(1).astype("float64")
        profile = src.profile.copy()

    power = np.where(power > 0, power, np.nan)
    db = 10 * np.log10(power)

    profile.update(dtype="float32", count=1)
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(db.astype("float32"), 1)
    print(f"Wrote {out_tif}")


# ---------------------------------------------------------------------------
# STAGE 4 -- co-register both sensors to one common grid over Demra ward
# ---------------------------------------------------------------------------
def stage4_coregister(src_tif: Path, out_tif: Path, ref_bounds=DEMRA_BBOX, ref_crs=TARGET_CRS, ref_res=TARGET_RESOLUTION):
    """Reproject/resample src_tif onto a shared UTM grid clipped to the Demra
    bounding box, so NISAR and Sentinel-1 pixels line up 1:1 for Phase 3 differencing.

    NOTE: this builds the destination transform directly from ref_bounds (the Demra
    bbox) rather than from the source raster's own extent -- calculate_default_transform
    only preserves the *source's* footprint, which would silently skip the ward clip
    entirely. Using an explicit dst_transform/width/height is what actually crops both
    sensors down to the same Demra window.
    """
    from rasterio.transform import from_origin

    minx, miny, maxx, maxy = ref_bounds
    width = int(round((maxx - minx) / ref_res))
    height = int(round((maxy - miny) / ref_res))
    transform = from_origin(minx, maxy, ref_res, ref_res)

    with rasterio.open(src_tif) as src:
        profile = src.profile.copy()
        profile.update(crs=ref_crs, transform=transform, width=width, height=height)

        out_tif.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_tif, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )
    print(f"Co-registered -> {out_tif}  ({width}x{height} px @ {ref_res}m, clipped to Demra bbox)")


# ---------------------------------------------------------------------------
# STAGE 5 -- mask permanent water & built-up using ESA WorldCover
# ---------------------------------------------------------------------------
def stage5_mask_water_builtup(backscatter_tif: Path, worldcover_tif: Path, out_tif: Path):
    """Zero out permanent-water and built-up pixels (per the brief's Phase 2
    masking step) so Phase 3's change detection only runs over vegetated /
    exposed land where the flood signal is ambiguous and worth comparing.
    worldcover_tif must already be reprojected/clipped to match backscatter_tif's grid.
    """
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


# ---------------------------------------------------------------------------
# STAGE 6 -- save final analysis-ready set (just a naming convention wrapper)
# ---------------------------------------------------------------------------
def stage6_save(name: str, data_tif: Path):
    final_dir = PROC_DIR / "analysis_ready"
    final_dir.mkdir(parents=True, exist_ok=True)
    dest = final_dir / f"{name}.tif"
    import shutil

    shutil.copy(data_tif, dest)
    print(f"Analysis-ready: {dest}")


if __name__ == "__main__":
    print(__doc__)
    print("Run stages individually once each input is ready -- see module docstring for order.")
