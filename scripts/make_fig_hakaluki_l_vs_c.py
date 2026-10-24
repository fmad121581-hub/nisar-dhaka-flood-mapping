"""
Figure + metrics: NISAR L-band (HH, HH-HV) vs Sentinel-1 C-band (VV, VV-VH) at Hakaluki Haor
through the 2026 recession. 22 Aug NISAR excluded (corrupt granule).

Metrics per sensor and class:
  plateau SD   -- date-to-date standard deviation while stage was at/above danger level
                  (to 13 Aug), i.e. how stable each sensor is when nothing should change;
  recession    -- mean of the last two dates minus the plateau mean (response to drainage);
  ratio change -- same for the co/cross ratio.
"""
from pathlib import Path
import csv
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(r"D:\NISAR_Dhaka_Flood")
RV = ROOT / "processed" / "reanalysis_v2"
OUT = ROOT / "figures" / "figure_v2_03_hakaluki_L_vs_C.png"
PLATEAU_END = "20260813"
CLASSES = [("cropland", "#b5651d"), ("grassland", "#7fb800"), ("wetland", "#2a9d8f"),
           ("tree cover", "#1b4332"), ("permanent water", "#1d4e89")]


def load(path, co, ratio, exclude=()):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["date"] in exclude:
            continue
        out.setdefault(r["cls"], {})[r["date"]] = (float(r[co]), float(r[ratio]))
    return out


nis = load(RV / "phase16_hakaluki_timeseries.csv", "hh_db", "hh_minus_hv_db", {"20260822"})
s1 = load(RV / "phase18_hakaluki_s1_timeseries.csv", "vv_db", "vv_minus_vh_db")
stage = {}
for r in csv.DictReader(open(RV / "fenchuganj_stage_2026.csv")):
    if r["stage_m_msl"]:
        stage.setdefault(r["stage_date"], float(r["stage_m_msl"]))

rows = []
print(f"{'class':16s} {'sensor':6s} {'plateau SD':>10s} {'recession':>10s} {'ratio chg':>10s}")
for cls, _ in CLASSES:
    for name, ser in (("NISAR", nis), ("S1", s1)):
        d = ser[cls]
        dates = sorted(d)
        plat = [d[x] for x in dates if x <= PLATEAU_END]
        last2 = [d[x] for x in dates[-2:]]
        sd = float(np.std([p[0] for p in plat], ddof=1))
        rec = float(np.mean([p[0] for p in last2]) - np.mean([p[0] for p in plat]))
        rch = float(np.mean([p[1] for p in last2]) - np.mean([p[1] for p in plat]))
        rows.append(dict(cls=cls, sensor=name, n_plateau=len(plat), plateau_sd_db=sd,
                         recession_change_db=rec, ratio_change_db=rch,
                         last_dates="+".join(dates[-2:])))
        print(f"{cls:16s} {name:6s} {sd:10.2f} {rec:+10.2f} {rch:+10.2f}")

with open(RV / "hakaluki_L_vs_C_metrics.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

fig, ax = plt.subplots(3, 2, figsize=(11, 8.5), sharex=True,
                       gridspec_kw={"height_ratios": [1, 2, 2]})
st = sorted(stage)
for c in (0, 1):
    ax[0, c].plot([dt.datetime.strptime(s, "%Y-%m-%d") for s in st], [stage[s] for s in st], "o-", color="#444")
    ax[0, c].axhline(9.45, ls="--", lw=1, color="#c0392b")
    ax[0, c].set_ylabel("Fenchuganj stage\n(m MSL)")
for c, (ser, title, co, rat) in enumerate(((nis, "NISAR L-band", "HH (dB)", "HH − HV (dB)"),
                                           (s1, "Sentinel-1 C-band (OPERA RTC)", "VV (dB)", "VV − VH (dB)"))):
    ax[0, c].set_title(title)
    for cls, col in CLASSES:
        d = ser[cls]
        ds = sorted(d)
        t = [dt.datetime.strptime(x, "%Y%m%d") for x in ds]
        ax[1, c].plot(t, [d[x][0] for x in ds], "o-", ms=4, color=col, label=cls)
        ax[2, c].plot(t, [d[x][1] for x in ds], "o-", ms=4, color=col, label=cls)
    ax[1, c].set_ylabel(co)
    ax[2, c].set_ylabel(rat)
    ax[1, c].set_ylim(-22, -2)
    ax[2, c].set_ylim(4.5, 12.5)
ax[1, 0].legend(fontsize=8, ncol=2, loc="center left")
for a in ax.ravel():
    a.grid(alpha=0.3)
    a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print(f"Wrote {OUT}")
