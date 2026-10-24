"""
Phase 5 follow-up -- does 20m class-mean averaging wash out a real but fragmented
flood signal? Re-run Experiment 1 (ascending: NISAR Jun30 dry -> Jul12 during) at
NISAR's *native* 10m grid instead of the 20m grid used for the main analysis
(20m was chosen there only to match Sentinel-1's coarser native resolution).

Both june30_asc and july12_asc GCOV files are natively 10m on this track (confirmed
by phase2/phase4 preprocessing logs), and ESA WorldCover is natively 10m too, so
this needs no resampling of the SAR data itself -- only a shared-grid crop.
"""

import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

_HOME = Path(os.environ["HOME"])
PROC_DIR = _HOME / "mnt" / "d" / "NISAR_Dhaka_Flood" / "processed"
WORLDCOVER_RAW = _HOME / "mnt" / "d" / "NISAR_Dhaka_Flood" / "raw_data" / "worldcover" / "ESA_WorldCover_10m_2020_v100_N21E090_Map.tif"

DEMRA_BBOX = (240673.90374, 2621657.94409, 246404.89660, 2630128.18446)  # EPSG:32646, meters
TARGET_CRS = "EPSG:32646"
RES_10M = 10

NISAR_PRE = PROC_DIR / "nisar_hv" / "june30_asc_HVHV.tif"
NISAR_DURING = PROC_DIR / "nisar_hv" / "july12_asc_dualpol_HV_dB.tif"

WC_CLASSES = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland",
              50: "built-up", 60: "bare/sparse", 80: "permanent water", 90: "wetland"}
VEGETATED = [10, 20, 30, 40, 90]


def coregister_10m(src_tif, out_tif, resampling=Resampling.bilinear):
    minx, miny, maxx, maxy = DEMRA_BBOX
    width = int(round((maxx - minx) / RES_10M))
    height = int(round((maxy - miny) / RES_10M))
    transform = from_origin(minx, maxy, RES_10M, RES_10M)

    with rasterio.open(src_tif) as src:
        profile = src.profile.copy()
        profile.update(crs=TARGET_CRS, transform=transform, width=width, height=height)
        out_tif.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_tif, "w", **profile) as dst:
            reproject(source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                      src_transform=src.transform, src_crs=src.crs, dst_transform=transform,
                      dst_crs=TARGET_CRS, resampling=resampling)
    return width, height


def read(path):
    with rasterio.open(path) as src:
        return src.read(1)


def main():
    out_dir = PROC_DIR / "native10m_check"
    pre_out = out_dir / "nisar_june30_asc_HVHV_10m.tif"
    during_out = out_dir / "nisar_july12_asc_HVHV_10m.tif"
    wc_out = out_dir / "worldcover_10m.tif"

    w, h = coregister_10m(NISAR_PRE, pre_out)
    coregister_10m(NISAR_DURING, during_out)
    coregister_10m(WORLDCOVER_RAW, wc_out, resampling=Resampling.nearest)
    print(f"Native-10m grid: {w}x{h} px (vs 287x424 at 20m -- should be ~2x2 as many pixels)")

    pre = read(pre_out)
    during = read(during_out)
    classes = read(wc_out).astype(int)
    delta = during - pre

    print("\n=== Per-class delta, 10m native grid (vs 20m result in phase5 doc) ===")
    for code in VEGETATED:
        name = WC_CLASSES[code]
        mask = classes == code
        vals = delta[mask]
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue
        frac_up = float(np.mean(vals > 3.0)) * 100
        frac_down = float(np.mean(vals < -3.0)) * 100
        print(f"  {name:16s} n={vals.size:6d}  mean={np.mean(vals):+.2f}dB  median={np.median(vals):+.2f}dB  "
              f"std={np.std(vals):.2f}  >+3dB:{frac_up:5.1f}%  <-3dB:{frac_down:5.1f}%")


if __name__ == "__main__":
    main()
