"""
Supplementary tables for manuscript v2 (written as CSV and collected in one markdown file).
  S1 granules used (NISAR GCOV, OPERA RTC-S1 dates/bursts, HyP3 RTC) with screening outcome
  S2 stage and lake-level records with sources
  S3 GCOV noise-equivalent backscatter over Dhaka (from calibrationInformation)
  S4 Hakaluki basin vs hill tree cover through the recession (phase 17)
  S5 class means with 95 % block-bootstrap CIs and water-like shares (phase 21)
"""
from pathlib import Path
import csv
import re
import numpy as np
import h5py

ROOT = Path(r"D:\NISAR_Dhaka_Flood")
RV = ROOT / "processed" / "reanalysis_v2"
OUT = ROOT / "manuscript_v2" / "supplement"
OUT.mkdir(exist_ok=True)
DL = Path(r"C:\Users\user\Downloads")
G = "science/LSAR/GCOV/grids/frequencyA/"


def write(name, rows):
    with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


def md(rows, cols=None):
    cols = cols or list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


# ---- S1 granules
USE = {"_141_A_014_": "Hakaluki", "_069_A_013_": "Kaptai"}
s1 = []
seen = set()
for p in sorted((ROOT / "raw_data" / "nisar_gcov").glob("*_001.h5")) + sorted(DL.glob("NISAR_L2_PR_GCOV_*_001.h5")):
    stem = p.name[:-3]
    if stem in seen:
        continue
    d = re.search(r"_(2026\d{4})T", p.name).group(1)
    site = next((v for k, v in USE.items() if k in p.name), None)
    if "_069_A_014_" in p.name and d in ("20260630", "20260712"):
        site = "Dhaka (Exp 1)"
    if "_091_D_077_" in p.name:
        site = "Dhaka (Exp 2)"
    if site is None:
        continue
    try:
        with h5py.File(p, "r") as f:
            full = f["science/LSAR/identification/isFullFrame"][()].decode()
            mixed = f["science/LSAR/identification/isMixedMode"][()].decode()
            mv = float(f[G + "HHHH"].attrs["mean_value"])
    except OSError:
        continue  # truncated duplicate copy
    seen.add(stem)
    verdict = "used"
    if d == "20260714":
        verdict = "excluded: partial frame, mixed-mode; mask = 0 over 96.7 % of Uttar Khan"
    if d == "20260702":
        verdict = "excluded: flagged mixed-mode (and its pair failed)"
    if d == "20260822":
        verdict = "excluded: corrupt radiometry (mean_value 586)"
    s1.append(dict(sensor="NISAR GCOV", site=site, date=d, granule=stem, full_frame=full, mixed_mode=mixed,
                   HHHH_mean_value=f"{mv:.3g}", outcome=verdict))
opera = {}
for p in sorted((ROOT / "raw_data" / "opera_rtc_s1").glob("*_VV.tif")):
    m = re.search(r"_T041-(\d+)-(IW\d)_(2026\d{4})T.*_(S1[A-D])_", p.name)
    b, iw, d, sat = m.groups()
    site = "Hakaluki" if b in ("086059", "086060") else "Kaptai"
    opera.setdefault((site, d, sat), []).append(f"{b}-{iw}")
for (site, d, sat), bursts in sorted(opera.items()):
    if site == "Hakaluki" or d in ("20260626", "20260708", "20260720", "20260801"):
        s1.append(dict(sensor=f"OPERA RTC-S1 ({sat})", site=site, date=d,
                       granule="T041 bursts " + ", ".join(sorted(set(bursts))), full_frame="", mixed_mode="",
                       HHHH_mean_value="", outcome="used"))
for d in sorted((ROOT / "processed" / "s1_rtc").glob("S1D_IW_20260701*")) + \
         sorted((ROOT / "processed" / "s1_rtc").glob("S1D_IW_20260713*")):  # ascending pair only
    if not d.is_dir():  # skip the delivered .zip next to the extracted folder
        continue
    s1.append(dict(sensor="Sentinel-1D HyP3 RTC", site="Dhaka", date=d.name[7:15], granule=d.name,
                   full_frame="", mixed_mode="", HHHH_mean_value="", outcome="used"))
write("tableS1_granules", s1)

# ---- S2 stage / lake level
s2 = [dict(site="Hakaluki (Fenchuganj, Kushiyara)", **r) for r in csv.DictReader(open(RV / "fenchuganj_stage_2026.csv"))]
KAPTAI = [("2026-06-10", 73.91, "https://bangla.dhakatribune.com/bangladesh/104123"),
          ("2026-07-08", 84.60, "https://www.tbsnews.net/bangladesh/kaptai-hydropower-plant-resumes-all-five-units-lake-water-level-rises-1483251"),
          ("2026-07-10", 90.31, "https://www.dhakatribune.com/bangladesh/power-energy/414781/kaptai-hydropower-plant-resumes-all-5-units-as"),
          ("2026-07-13", 100.60, "https://www.dinajpurtv.com/country/27670"),
          ("2026-07-14", 102.25, "https://www.bssnews.net/others/405513"),
          ("2026-07-18", 104.08, "https://www.bssnews.net/others/406862"),
          ("2026-07-22", 103.31, "https://www.ittefaq.com.bd/800022/")]
for d, ft, url in KAPTAI:
    s2.append(dict(site="Kaptai Lake (BPDB via press)", nisar_date="", stage_date=d,
                   stage_m_msl=f"{ft * 0.3048:.2f}", cm_vs_danger_9p45=f"{ft:.2f} ft",
                   quality="news report of BPDB reading", source=url))
write("tableS2_stage_and_levels", s2)

# ---- S3 NESZ over Dhaka
B = {"Demra": (240674, 2621658, 246405, 2630128), "Uttar Khan": (236583, 2639492, 242586, 2645682)}
s3 = []
for pat in ("*20260630T*", "*20260702T*", "*20260712T*", "*20260714T*"):
    p = next(x for x in (ROOT / "raw_data" / "nisar_gcov").glob(pat + "_001.h5"))
    with h5py.File(p, "r") as f:
        nb = "science/LSAR/GCOV/metadata/calibrationInformation/frequencyA/noiseEquivalentBackscatter/"
        x, y = f[nb + "xCoordinates"][:], f[nb + "yCoordinates"][:]
        for site, (x0, y0, x1, y1) in B.items():
            row = dict(date=re.search(r"_(2026\d{4})T", p.name).group(1), mode=p.name.split("_")[9], site=site)
            for pol in ("HH", "HV"):
                v = f[nb + pol][:][np.ix_((y >= y0 - 2000) & (y <= y1 + 2000), (x >= x0 - 2000) & (x <= x1 + 2000))]
                row[f"NESZ_{pol}_dB"] = f"{10 * np.log10(np.median(v[v > 0])):.2f}"
            s3.append(row)
write("tableS3_nesz_dhaka", s3)

# ---- S4, S5 from processed outputs
s4 = write("tableS4_hakaluki_basin_vs_hill", list(csv.DictReader(open(RV / "phase17_hakaluki_lowland_upland.csv"))))
s5 = write("tableS5_class_means_ci", list(csv.DictReader(open(RV / "phase21_thresholds_ci.csv"))))

with open(OUT / "supplement_tables.md", "w", encoding="utf-8") as fh:
    fh.write("# Supplementary tables (v2 draft)\n\n")
    fh.write("## Table S1. Granules used and screening outcome\n\n" + md(s1) + "\n\n")
    fh.write("## Table S2. Stage and lake-level records\n\n" + md(s2) + "\n\n")
    fh.write("## Table S3. GCOV noise-equivalent backscatter over the Dhaka wards\n\n" + md(s3) + "\n\n")
    fh.write("## Table S4. Hakaluki tree cover, basin (< 12 m) vs hills (> 40 m)\n\n" + md(s4) + "\n\n")
    fh.write("## Table S5. Class means, 95 % block-bootstrap CIs and water-like shares\n\n" + md(s5) + "\n")
print(f"S1 {len(s1)} rows, S2 {len(s2)}, S3 {len(s3)}, S4 {len(s4)}, S5 {len(s5)} -> {OUT}")
