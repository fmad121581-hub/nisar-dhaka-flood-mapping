"""
Figures 4 and 5 (v2).

Figure 4 -- Kaptai Lake: (a) reported lake level (BPDB via press) with SAR dates marked;
(b) water-like share per class, NISAR HH and Sentinel-1 VV, each on its own pooled Otsu
threshold (phase 21), Sentinel-1 restricted to the NISAR footprint.

Figure 5 -- Dhaka: (a) class change relative to built-up land, NISAR HH and S1 VV, ascending
peak-rain pair (the descending pair failed screening); (b) built-up co-pol change
by elevation band across central Dhaka (phase 15).
"""
from pathlib import Path
import csv
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(r"D:\NISAR_Dhaka_Flood")
RV = ROOT / "processed" / "reanalysis_v2"

# ---------------- Figure 5: Kaptai
LEVEL_FT = {"2026-06-10": 73.91, "2026-07-08": 84.60, "2026-07-10": 90.31, "2026-07-13": 100.60,
            "2026-07-14": 102.25, "2026-07-16": 103.91, "2026-07-18": 104.08, "2026-07-19": 103.80,
            "2026-07-22": 103.31, "2026-08-29": 105.64}
rows = [r for r in csv.DictReader(open(RV / "phase21_thresholds_ci.csv")) if r["site"] == "Kaptai"]
fig, ax = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1.6]})
d = sorted(LEVEL_FT)
ax[0].plot([dt.datetime.fromisoformat(x) for x in d], [LEVEL_FT[x] * 0.3048 for x in d], "o-", color="#444")
ax[0].set_ylabel("Kaptai Lake level\n(m MSL)")
ax[0].axvspan(dt.datetime(2026, 7, 18), dt.datetime(2026, 7, 22, 23), color="#bbb", alpha=0.4, lw=0)
ax[0].text(dt.datetime(2026, 7, 18, 12), 23.0, "spillway\nopen", fontsize=7.5)
ax[0].text(0.01, 0.95, "(a)", transform=ax[0].transAxes, va="top", fontweight="bold")
for cls, col in (("wetland", "#2a9d8f"), ("grassland", "#7fb800"), ("cropland", "#b5651d")):
    for sensor, ls, mk in (("NISAR", "-", "o"), ("S1", "--", "s")):
        sub = sorted((r["date"], float(r["pct_water_like"])) for r in rows if r["cls"] == cls and r["sensor"] == sensor)
        ax[1].plot([dt.datetime.strptime(x, "%Y%m%d") for x, _ in sub], [v for _, v in sub], ls=ls, marker=mk,
                   color=col, label=f"{cls}, {'NISAR HH' if sensor == 'NISAR' else 'Sentinel-1 VV'}")
ax[1].set_ylabel("water-like share (%)\n(own Otsu threshold)")
ax[1].set_ylim(0, 100)
ax[1].legend(fontsize=7.5, ncol=2, loc="lower right")
ax[1].text(0.01, 0.97, "(b)", transform=ax[1].transAxes, va="top", fontweight="bold")
for a in ax:
    a.grid(alpha=0.3)
    a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
fig.tight_layout()
fig.savefig(ROOT / "figures" / "figure_v2_04_kaptai.png", dpi=200)

# ---------------- Figure 6: Dhaka
n13 = list(csv.DictReader(open(RV / "phase13_results.csv")))
s14 = list(csv.DictReader(open(RV / "phase14_s1_results.csv")))
CL = ["tree cover", "grassland", "cropland", "wetland", "permanent water"]


def rel(rows, site, exp, pol, masked=True):
    sel = [r for r in rows if r["site"] == site and r["experiment"] == exp and r["pol"] == pol
           and (r.get("mask", "masked") == "masked")]
    m = {r["cls"]: float(r["mean_db"]) if r["mean_db"] not in ("", "nan") else np.nan for r in sel}
    return [m.get(c, np.nan) - m.get("built-up", np.nan) for c in CL], m.get("built-up", np.nan)


fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
# the descending pair (Exp 2) failed granule screening and is not shown
series = [("Demra", "Demra", "Exp1_asc"), ("Uttar Khan", "UttarKhan", "Exp1_asc")]
w = 0.18
xs = np.arange(len(CL))
k = 0
for label, site, exp in series:
    for sensor, rows_, pol, hatch in (("NISAR HH", n13, "HH", None), ("S1 VV", s14, "VV", "///")):
        v, _ = rel(rows_, site, exp, pol)
        ax[0].bar(xs + (k - 1.5) * w, v, w, label=f"{label}, {sensor}", hatch=hatch,
                  color=["#4c72b0", "#dd8452", "#55a868"][k // 2], edgecolor="black", lw=0.4)
        k += 1
ax[0].axhline(0, color="black", lw=0.8)
ax[0].set_xticks(xs, CL, fontsize=8)
ax[0].set_ylabel("change relative to built-up (dB)")
ax[0].legend(fontsize=7, ncol=2)
ax[0].text(0.01, 0.97, "(a)", transform=ax[0].transAxes, va="top", fontweight="bold")
ax[0].grid(alpha=0.3, axis="y")

el = list(csv.DictReader(open(RV / "phase15_builtup_by_elevation.csv")))
for name, col in (("NISAR HH 30Jun->12Jul", "#4c72b0"), ("S1 VV 01Jul->13Jul", "#dd8452")):
    sub = [r for r in el if r["delta"] == name]
    lab = [f"{int(float(r['dem_lo']))}–{int(float(r['dem_hi']))}" if float(r["dem_hi"]) < 100
           else f">{int(float(r['dem_lo']))}" for r in sub]
    lab[0] = "<4"
    ax[1].plot(lab, [float(r["mean_db"]) for r in sub], "o-", color=col, label=name.replace("->", "→"))
ax[1].set_xlabel("Copernicus DSM height band (m)")
ax[1].set_ylabel("built-up co-pol change (dB)")
ax[1].set_ylim(0, 2)
ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3)
ax[1].text(0.01, 0.97, "(b)", transform=ax[1].transAxes, va="top", fontweight="bold")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "figure_v2_05_dhaka.png", dpi=200)
print("Wrote figure_v2_04_kaptai.png and figure_v2_05_dhaka.png")
