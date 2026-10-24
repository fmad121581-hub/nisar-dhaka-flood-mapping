"""
Phase 22 -- Sensitivity of the Hakaluki stage-backscatter correlations to which stage reading
is matched to each NISAR date.

For every NISAR date with a stage value, all readings found within about a week either side
are listed (ALTERNATIVES, values in m MSL, sources in Table S2 / REVISION_NOTES). Spearman's
rho between stage and cropland HH (and HH-HV) is recomputed for every combination.
Readings can tie or swap rank between neighbouring dates (e.g. 27 Sep bracketed by 8.67 m on
20 Sep and 7.20 m on 1 Oct, against 7.21 m on 8 Oct), which is why rho varies.

Outputs: processed/reanalysis_v2/phase22_stage_alternatives.csv, phase22_stage_sensitivity.csv
"""
from pathlib import Path
import csv
import itertools
from scipy.stats import spearmanr

ROOT = Path(r"D:\NISAR_Dhaka_Flood")
RV = ROOT / "processed" / "reanalysis_v2"

# nisar date -> [(reading date, stage m MSL, source type)]
ALTERNATIVES = {
    "20260623": [("2026-06-27", 9.46, "news report of FFWC bulletin"),
                 ("2026-06-28", 9.49, "news report of FFWC bulletin")],
    "20260717": [("2026-07-14", 9.71, "news report of FFWC bulletin"),
                 ("2026-07-18", 9.80, "news report of FFWC bulletin")],
    "20260810": [("2026-08-08", 9.63, "secondary compilation"),
                 ("2026-08-11", 9.67, "news report of FFWC bulletin")],
    "20260903": [("2026-08-24", 9.24, "secondary compilation"),
                 ("2026-08-29", 9.11, "FFWC bulletin, web archive copy"),
                 ("2026-08-30", 9.22, "FFWC bulletin, web archive copy")],
    "20260915": [("2026-09-17", 8.94, "secondary compilation")],
    "20260927": [("2026-09-20", 8.67, "search-engine copy of FFWC bulletin"),
                 ("2026-10-01", 7.20, "secondary compilation")],
    "20261009": [("2026-10-01", 7.20, "secondary compilation"),
                 ("2026-10-08", 7.21, "secondary compilation")],
}
MAIN = {"20260623": 9.49, "20260717": 9.71, "20260810": 9.67, "20260903": 9.22,
        "20260915": 8.94, "20260927": 8.67, "20261009": 7.21}

crop = {r["date"]: r for r in csv.DictReader(open(RV / "phase16_hakaluki_timeseries.csv")) if r["cls"] == "cropland"}
dates = sorted(ALTERNATIVES)

with open(RV / "phase22_stage_alternatives.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["nisar_date", "reading_date", "stage_m_msl", "source_type"])
    for d in dates:
        for rd, v, src in ALTERNATIVES[d]:
            w.writerow([d, rd, v, src])

rows = []
for var in ("hh_db", "hh_minus_hv_db"):
    y = [float(crop[d][var]) for d in dates]
    main = spearmanr([MAIN[d] for d in dates], y)[0]
    rhos = [spearmanr(c, y)[0] for c in itertools.product(*[[a[1] for a in ALTERNATIVES[d]] for d in dates])]
    rows.append(dict(variable=var, rho_main=round(main, 3), n_combinations=len(rhos),
                     rho_min=round(min(rhos), 3), rho_max=round(max(rhos), 3)))
    print(f"{var}: main rho {main:.2f}; {len(rhos)} combinations: {min(rhos):.2f} to {max(rhos):.2f}")
with open(RV / "phase22_stage_sensitivity.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
