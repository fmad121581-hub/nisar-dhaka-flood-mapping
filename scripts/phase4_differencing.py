"""
Phase 4 -- Two-experiment differencing + absolute-backscatter verification
Runs entirely on the already-produced analysis_ready/*.tif rasters (8 total:
4 from Phase 2/3, 4 new dry-baseline ones from this phase). All 8 share the
identical grid: 287x424 px, 20m, EPSG:32646, clipped to the Demra bbox --
confirmed by phase2/phase4 preprocessing logs -- so no further alignment
needed here.

Outputs:
  1. Delta (during - pre) rasters for both experiments, both sensors.
  2. Per-WorldCover-class mean/median delta, both experiments, both sensors.
  3. Absolute (not delta) backscatter stats per class on the four "during"
     scenes, benchmarked against each raster's own permanent-water class as
     an empirical low-backscatter reference for "looks like open water."
"""

import os
from pathlib import Path

import numpy as np
import rasterio

_HOME = Path(os.environ["HOME"])
PROC_DIR = _HOME / "mnt" / "d" / "NISAR_Dhaka_Flood" / "processed"
AR_DIR = PROC_DIR / "analysis_ready"
COREG_DIR = PROC_DIR / "coregistered"  # pre-mask -- needed for the permanent-water reference,
                                        # since analysis_ready/* already has water masked to NaN
DIFF_DIR = PROC_DIR / "differenced"
WORLDCOVER_CLIPPED = PROC_DIR / "worldcover_clipped.tif"

# analysis_ready/*.tif filename -> matching pre-mask coregistered/*.tif filename
COREG_NAME = {
    "nisar_june30_asc_HVHV.tif": "nisar_june30_asc_HVHV_coreg.tif",
    "nisar_july2_desc_HVHV.tif": "nisar_july2_desc_HVHV_coreg.tif",
    "nisar_july12_asc_HVHV.tif": "nisar_july12_asc_HVHV_coreg.tif",
    "nisar_july14_desc_HVHV.tif": "nisar_july14_desc_HVHV_coreg.tif",
    "s1_july1_asc_VH.tif": "s1_july1_asc_VH_coreg.tif",
    "s1_july3_desc_VH.tif": "s1_july3_desc_VH_coreg.tif",
    "s1_july13_asc_VH.tif": "s1_july13_asc_VH_coreg.tif",
    "s1_july15_desc_VH.tif": "s1_july15_desc_VH_coreg.tif",
}

# ESA WorldCover class codes present over Demra (per phase2-preprocessing-status.md)
WC_CLASSES = {
    10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland",
    50: "built-up", 60: "bare/sparse", 80: "permanent water", 90: "wetland",
}
VEGETATED = [10, 20, 30, 40, 90]  # tree, shrub, grassland, cropland, wetland -- the classes Phase 2 masked OUT of the delta comparison, but we want their ABSOLUTE values here

EXPERIMENTS = {
    "exp1_peak_ascending": {
        "nisar_pre": "nisar_june30_asc_HVHV.tif", "nisar_during": "nisar_july12_asc_HVHV.tif",
        "s1_pre": "s1_july1_asc_VH.tif", "s1_during": "s1_july13_asc_VH.tif",
    },
    "exp2_recession_descending": {
        "nisar_pre": "nisar_july2_desc_HVHV.tif", "nisar_during": "nisar_july14_desc_HVHV.tif",
        "s1_pre": "s1_july3_desc_VH.tif", "s1_during": "s1_july15_desc_VH.tif",
    },
}


def read(path):
    with rasterio.open(path) as src:
        return src.read(1), src.profile.copy()


def write_like(data, profile, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = profile.copy()
    profile.update(dtype="float32", count=1)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)


def class_stats(data, classes, class_code):
    mask = classes == class_code
    vals = data[mask]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return None
    return {
        "n": int(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
        "std": float(np.std(vals)),
    }


def main():
    classes, _ = read(WORLDCOVER_CLIPPED)
    classes = classes.astype(int)

    print("=" * 80)
    print("PART 1 -- Delta (during - pre) rasters + per-class stats")
    print("=" * 80)
    for exp_name, cfg in EXPERIMENTS.items():
        print(f"\n--- {exp_name} ---")
        for sensor in ("nisar", "s1"):
            pre, profile = read(AR_DIR / cfg[f"{sensor}_pre"])
            during, _ = read(AR_DIR / cfg[f"{sensor}_during"])
            delta = during - pre  # dB - dB, already log-domain so this is the ratio in dB
            out_path = DIFF_DIR / f"delta_{sensor}_{exp_name}.tif"
            write_like(delta, profile, out_path)
            print(f"\n  {sensor.upper()} delta ({cfg[f'{sensor}_during']} - {cfg[f'{sensor}_pre']}):")
            for code, name in WC_CLASSES.items():
                if code in (50, 80):  # built-up + water masked out of the comparison per Phase 2 convention
                    continue
                s = class_stats(delta, classes, code)
                if s:
                    print(f"    {name:16s} n={s['n']:5d}  mean={s['mean']:+.2f}dB  median={s['median']:+.2f}dB  "
                          f"p10={s['p10']:+.2f} p90={s['p90']:+.2f}  std={s['std']:.2f}")

    print("\n" + "=" * 80)
    print("PART 1.5 -- Pixel-level anomaly fractions (is a *subset* of pixels flooded, even if the class mean isn't?)")
    print("=" * 80)
    DOUBLE_BOUNCE_THRESHOLD = 3.0   # dB increase -- typical reported magnitude for cross-pol enhancement under flooded vegetation
    SPECULAR_DROP_THRESHOLD = -3.0  # dB decrease -- candidate open-water-like signature (no double bounce, just smooth flooded surface)
    for exp_name, cfg in EXPERIMENTS.items():
        print(f"\n--- {exp_name} ---")
        for sensor in ("nisar", "s1"):
            pre, _ = read(AR_DIR / cfg[f"{sensor}_pre"])
            during, _ = read(AR_DIR / cfg[f"{sensor}_during"])
            delta = during - pre
            print(f"\n  {sensor.upper()}: fraction of pixels with |delta| > 3dB, by class")
            for code in VEGETATED:
                name = WC_CLASSES[code]
                mask = classes == code
                vals = delta[mask]
                vals = vals[~np.isnan(vals)]
                if vals.size == 0:
                    continue
                frac_up = float(np.mean(vals > DOUBLE_BOUNCE_THRESHOLD)) * 100
                frac_down = float(np.mean(vals < SPECULAR_DROP_THRESHOLD)) * 100
                print(f"    {name:16s} n={vals.size:5d}  >+3dB (candidate double-bounce): {frac_up:5.1f}%   "
                      f"<-3dB (candidate specular/open-water): {frac_down:5.1f}%")

    print("\n" + "=" * 80)
    print("PART 2 -- Absolute backscatter on 'during' scenes vs each raster's own water class")
    print("=" * 80)
    for exp_name, cfg in EXPERIMENTS.items():
        print(f"\n--- {exp_name} ---")
        for sensor in ("nisar", "s1"):
            during, _ = read(AR_DIR / cfg[f"{sensor}_during"])
            during_unmasked, _ = read(COREG_DIR / COREG_NAME[cfg[f"{sensor}_during"]])
            water_stats = class_stats(during_unmasked, classes, 80)
            print(f"\n  {sensor.upper()} {cfg[f'{sensor}_during']} -- absolute dB:")
            if water_stats:
                print(f"    [reference] permanent water   mean={water_stats['mean']:+.2f}dB  "
                      f"median={water_stats['median']:+.2f}dB  (n={water_stats['n']})")
            else:
                print("    [reference] permanent water   -- no valid pixels in this window")
            for code in VEGETATED:
                name = WC_CLASSES[code]
                s = class_stats(during, classes, code)
                if s:
                    gap = (s['mean'] - water_stats['mean']) if water_stats else float('nan')
                    print(f"    {name:16s} mean={s['mean']:+.2f}dB  median={s['median']:+.2f}dB  "
                          f"p10={s['p10']:+.2f}  n={s['n']:5d}  (vs water: {gap:+.2f}dB)")

    print("\n" + "=" * 80)
    print("PART 3 -- Same absolute check on 'pre' (dry) scenes, as a baseline for comparison")
    print("=" * 80)
    for exp_name, cfg in EXPERIMENTS.items():
        print(f"\n--- {exp_name} ---")
        for sensor in ("nisar", "s1"):
            pre, _ = read(AR_DIR / cfg[f"{sensor}_pre"])
            pre_unmasked, _ = read(COREG_DIR / COREG_NAME[cfg[f"{sensor}_pre"]])
            water_stats = class_stats(pre_unmasked, classes, 80)
            print(f"\n  {sensor.upper()} {cfg[f'{sensor}_pre']} (dry) -- absolute dB:")
            if water_stats:
                print(f"    [reference] permanent water   mean={water_stats['mean']:+.2f}dB  (n={water_stats['n']})")
            for code in VEGETATED:
                name = WC_CLASSES[code]
                s = class_stats(pre, classes, code)
                if s:
                    print(f"    {name:16s} mean={s['mean']:+.2f}dB  median={s['median']:+.2f}dB  n={s['n']:5d}")


if __name__ == "__main__":
    main()
