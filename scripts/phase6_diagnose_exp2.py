"""
Diagnostic: Exp2 NISAR (july2_desc -> july14_desc) delta at Uttar Khan is enormous
(-5 to -7dB mean, std up to 7.3dB) -- 5-7x anything seen anywhere else in this study,
and the SAME scene pair over Demra gave near-zero deltas (see phase5 doc: wetland
-0.86dB was the biggest, others <0.6dB). Before treating this as a flood signal,
check whether it's a genuine backscatter change or a calibration/registration
artifact specific to the Uttar Khan crop window.

Checks:
  1. Absolute (not delta) backscatter, pre vs during, at Uttar Khan, referenced to
     that scene's own permanent-water class (same technique as phase4_differencing
     Part 2/3 for Demra).
  2. Re-derive full stats (incl. std) for the SAME Exp2 NISAR pair over the ORIGINAL
     Demra bbox/grid, for direct side-by-side comparison of variance and absolute
     levels between the two locations.
"""
import os
from pathlib import Path

import numpy as np
import rasterio

_HOME = Path(os.environ["HOME"])
D_ROOT = _HOME / "mnt" / "d" / "NISAR_Dhaka_Flood"
PROC_DIR = D_ROOT / "processed"

UK_COREG = PROC_DIR / "uttarkhan_case" / "coregistered"
DEMRA_COREG = PROC_DIR / "coregistered"
DEMRA_WC = PROC_DIR / "worldcover_clipped.tif"

WC_CLASSES = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland",
              50: "built-up", 60: "bare/sparse", 80: "permanent water", 90: "wetland"}
VEGETATED = [10, 20, 30, 40, 90]


def read(p):
    with rasterio.open(p) as src:
        return src.read(1)


def stats(vals):
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return None
    return {"n": int(vals.size), "mean": float(np.mean(vals)), "median": float(np.median(vals)),
            "std": float(np.std(vals)), "p10": float(np.percentile(vals, 10)), "p90": float(np.percentile(vals, 90))}


def report_absolute(label, classes, pre, during):
    water_pre = stats(pre[classes == 80])
    water_during = stats(during[classes == 80])
    print(f"\n  -- {label} --")
    print(f"  water  pre:    mean={water_pre['mean']:+.2f}dB  n={water_pre['n']}" if water_pre else "  water pre: no pixels")
    print(f"  water  during: mean={water_during['mean']:+.2f}dB  n={water_during['n']}" if water_during else "  water during: no pixels")
    for code in VEGETATED:
        name = WC_CLASSES[code]
        mask = classes == code
        sp = stats(pre[mask])
        sd = stats(during[mask])
        if sp and sd:
            print(f"  {name:16s} PRE  mean={sp['mean']:+7.2f}dB median={sp['median']:+7.2f}dB std={sp['std']:5.2f}  n={sp['n']}")
            print(f"  {name:16s} DUR  mean={sd['mean']:+7.2f}dB median={sd['median']:+7.2f}dB std={sd['std']:5.2f}  n={sd['n']}")
            if water_during:
                print(f"  {name:16s} DUR vs water: {sd['mean'] - water_during['mean']:+.2f}dB")


def main():
    print("=" * 80)
    print("PART A -- Uttar Khan, Exp2 NISAR (july2_desc pre -> july14_desc during), absolute dB")
    print("=" * 80)
    classes_uk = read(UK_COREG / "worldcover_coreg.tif").astype(int)
    pre_uk = read(UK_COREG / "nisar_july2_desc_coreg.tif")
    during_uk = read(UK_COREG / "nisar_july14_desc_coreg.tif")
    report_absolute("Uttar Khan", classes_uk, pre_uk, during_uk)

    print("\n" + "=" * 80)
    print("PART B -- Demra, SAME Exp2 NISAR pair, absolute dB + full stats incl std (for comparison)")
    print("=" * 80)
    classes_dm = read(DEMRA_WC).astype(int)
    pre_dm = read(DEMRA_COREG / "nisar_july2_desc_HVHV_coreg.tif")
    during_dm = read(DEMRA_COREG / "nisar_july14_desc_HVHV_coreg.tif")
    report_absolute("Demra", classes_dm, pre_dm, during_dm)

    print("\n" + "=" * 80)
    print("PART C -- Demra delta std (was only mean/median in phase5 doc -- checking if std was already large there too)")
    print("=" * 80)
    delta_dm = during_dm - pre_dm
    for code in VEGETATED:
        name = WC_CLASSES[code]
        vals = delta_dm[classes_dm == code]
        s = stats(vals)
        if s:
            print(f"  {name:16s} mean={s['mean']:+.2f}dB  median={s['median']:+.2f}dB  std={s['std']:.2f}  n={s['n']}")


if __name__ == "__main__":
    main()

# --- Part D appended: Exp1 (mode-consistent DHDH_A/DHDH_A) water sanity check at Uttar Khan ---
def part_d():
    print("\n" + "=" * 80)
    print("PART D -- Uttar Khan, Exp1 NISAR (june30_asc DHDH_A -> july12_asc DHDH_A), water sanity check")
    print("=" * 80)
    classes_uk = read(UK_COREG / "worldcover_coreg.tif").astype(int)
    pre = read(UK_COREG / "nisar_june30_asc_coreg.tif")
    during = read(UK_COREG / "nisar_july12_asc_coreg.tif")
    wp = stats(pre[classes_uk == 80])
    wd = stats(during[classes_uk == 80])
    print(f"  water pre  (june30, DHDH_A): mean={wp['mean']:+.2f}dB n={wp['n']}" if wp else "no pixels")
    print(f"  water during (july12, DHDH_A): mean={wd['mean']:+.2f}dB n={wd['n']}" if wd else "no pixels")
    if wp and wd:
        print(f"  water shift: {wd['mean']-wp['mean']:+.2f}dB  (Exp2's same-location water shift was -8.00dB)")

part_d()
