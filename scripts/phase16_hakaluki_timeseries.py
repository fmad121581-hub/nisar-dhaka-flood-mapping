"""
Phase 16 -- Hakaluki Haor positive-control time series (NISAR track 141 frame 014, DHDH only).

Design: one acquisition mode, one geometry, every 12 days from 23 Jun to 9 Oct 2026, read
against the Fenchuganj (Kushiyara) stage record. As the haor recedes and standing water drops
below the height of the emergent vegetation, co-pol (HH) double-bounce should first rise and
then fall in vegetated classes, while open water stays dark. Absolute class means are reported
per date (no single pre/post pair), with the GCOV valid-sample mask applied and every granule
integrity-checked (full-frame flag, truncation).

Outputs: processed/reanalysis_v2/phase16_hakaluki_timeseries.csv
"""
from pathlib import Path
import csv
import re
import numpy as np
import h5py
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

ROOT = Path(r"D:\NISAR_Dhaka_Flood")
SEARCH_DIRS = [ROOT / "raw_data" / "nisar_gcov", Path(r"C:\Users\user\Downloads")]
WC_TILE = ROOT / "raw_data" / "worldcover" / "ESA_WorldCover_10m_2020_v100_N24E090_Map.tif"
OUT = ROOT / "processed" / "reanalysis_v2" / "phase16_hakaluki_timeseries.csv"

HAKALUKI_BBOX = (398749.23462869856, 2719084.5651329057, 412351.523585242, 2735786.232745403)
CRS = "EPSG:32646"
GRID = "science/LSAR/GCOV/grids/frequencyA/"
WC_CLASSES = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland",
              50: "built-up", 60: "bare/sparse", 80: "permanent water", 90: "wetland"}
REPORT = [10, 30, 40, 90, 80]


def granules():
    found = {}
    for d in SEARCH_DIRS:
        for p in d.glob("NISAR_L2_PR_GCOV_*_141_A_014_4005_DHDH_*.h5"):
            if p.name.endswith("_QA_STATS.h5"):
                continue
            date = re.search(r"_(2026\d{4})T", p.name).group(1)
            found.setdefault(date, []).append(p)
    out = {}
    for date, paths in sorted(found.items()):
        for p in paths:  # some copies are truncated downloads; keep the first that opens
            try:
                h5py.File(p, "r").close()
                out[date] = p
                break
            except OSError:
                print(f"{date}: truncated copy ignored: {p}")
    return out


def read(path):
    x0, y0, x1, y1 = HAKALUKI_BBOX
    with h5py.File(path, "r") as f:
        full = f["science/LSAR/identification/isFullFrame"][()].decode()
        x = f[GRID + "xCoordinates"][:]
        y = f[GRID + "yCoordinates"][:]
        xi = np.where((x >= x0) & (x <= x1))[0]
        yi = np.where((y >= y0) & (y <= y1))[0]
        sl = (slice(yi.min(), yi.max() + 1), slice(xi.min(), xi.max() + 1))
        hh = f[GRID + "HHHH"][sl]
        hv = f[GRID + "HVHV"][sl]
        m = f[GRID + "mask"][sl]
        dx, dy = x[1] - x[0], y[1] - y[0]
        t = from_origin(x[xi.min()] - dx / 2, y[yi.min()] - dy / 2, dx, -dy)
    valid = (m > 0) & (m < 255) & (hh > 0) & (hv > 0)
    return hh, hv, valid, t, full


rows = []
wc = None
ref_shape = None
for date, path in granules().items():
    try:
        hh, hv, valid, t, full = read(path)
    except OSError as e:  # truncated download
        print(f"{date}: SKIP, unreadable ({e})")
        continue
    if wc is None:
        ref_shape, ref_t = hh.shape, t
        wc = np.zeros(ref_shape, dtype=np.uint8)
        with rasterio.open(WC_TILE) as src:
            reproject(rasterio.band(src, 1), wc, dst_transform=t, dst_crs=CRS,
                      resampling=Resampling.nearest)
    assert hh.shape == ref_shape and t == ref_t, f"{date}: grid differs from first date"
    print(f"\n{date}  full_frame={full}  valid {valid.mean():.1%}  ({path.parent.name})")
    for cls in REPORT:
        sel = valid & (wc == cls)
        h = 10 * np.log10(hh[sel].mean())   # class mean of linear power, in dB
        v = 10 * np.log10(hv[sel].mean())
        ratio = h - v
        frac = float((10 * np.log10(hh[sel]) > -5.9).mean() * 100)  # Chapman et al. 2015 flooded-veg level
        rows.append(dict(date=date, cls=WC_CLASSES[cls], n=int(sel.sum()), hh_db=h, hv_db=v,
                         hh_minus_hv_db=ratio, pct_hh_gt_minus5p9=frac, full_frame=full))
        print(f"  {WC_CLASSES[cls]:16s} HH {h:6.2f}  HV {v:6.2f}  HH-HV {ratio:5.2f}  "
              f"HH>-5.9dB {frac:5.1f}%  n={sel.sum():,}")

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nWrote {OUT}")
