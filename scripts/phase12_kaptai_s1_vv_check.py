"""
Kaptai Lake -- S1 VV co-pol cross-check (secondary, S1-side only).

NISAR's raw .h5 granules for this site were already deleted after HVHV-only
extraction (per the disk-management policy), so a full dual-sensor co-pol
check would require re-downloading ~12GB of NISAR data. Since Kaptai Lake
is a calibration-replication site (not core flood evidence), that cost
isn't justified here. This script instead runs the S1-side VV check only,
to see whether the unusually large S1 VH per-class deltas already found
are a broad, non-selective scene-level shift (as at Hakaluki Haor) or
genuinely class-selective, using the same permanent-water-calibration
framing as every other site.
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/mnt/d/NISAR_Dhaka_Flood/scripts"))
from pathlib import Path
import numpy as np
import rasterio
import phase12_kaptai_lake as k

DOWNLOADS = k.DOWNLOADS
OUT_DIR = k.OUT_DIR
raw_dir = OUT_DIR / "raw_extract"
coreg_dir = OUT_DIR / "coregistered"

S1_VV_SOURCES = {
    "pre": k._vsizip("S1D_IW_20260626T115524_DVP_RTC20_G_gpuned_8667.zip", "S1D_IW_20260626T115524_DVP_RTC20_G_gpuned_8667_VV.tif"),
    "during": k._vsizip("S1D_IW_20260720T115525_DVP_RTC20_G_gpuned_DA78.zip", "S1D_IW_20260720T115525_DVP_RTC20_G_gpuned_DA78_VV.tif"),
}

print("=== Step 1: extract S1 VV rasters (windowed reads) ===")
for label, path in S1_VV_SOURCES.items():
    k.extract_s1(path, raw_dir / f"s1vv_{label}.tif")

print("\n=== Step 2: co-register to shared 20m grid ===")
for label in S1_VV_SOURCES:
    k.coregister(raw_dir / f"s1vv_{label}.tif", coreg_dir / f"s1vv_{label}_coreg.tif")

classes = k.read(coreg_dir / "worldcover_coreg.tif").astype(int)

print("\n=== Step 3: permanent-water calibration check (S1 VV) ===")
water_mask = classes == 80
pre = k.read(coreg_dir / "s1vv_pre_coreg.tif")
during = k.read(coreg_dir / "s1vv_during_coreg.tif")
delta = during - pre
vals = delta[water_mask]
vals = vals[~np.isnan(vals)]
shift = float(np.mean(vals))
print(f"  S1 VV: permanent-water shift (during - pre) = {shift:+.2f} dB (n={vals.size})")

print("\n=== Step 4: S1 VV delta (during - pre), per vegetated class ===")
for code in k.VEGETATED:
    name = k.WC_CLASSES[code]
    mask = classes == code
    v = delta[mask]
    s = k.class_stats(v)
    if s:
        v2 = v[~np.isnan(v)]
        frac_up = float(np.mean(v2 > 3.0)) * 100
        frac_down = float(np.mean(v2 < -3.0)) * 100
        print(f"    {name:16s} n={s['n']:7d}  mean={s['mean']:+.2f}dB  median={s['median']:+.2f}dB  "
              f"std={s['std']:.2f}  >+3dB:{frac_up:5.1f}%  <-3dB:{frac_down:5.1f}%")
