"""
Phase 18 -- Sentinel-1 (C-band) counterpart of phase 16 at Hakaluki Haor, from OPERA RTC-S1
(30 m, gamma0 linear power), relative orbit 41 ascending, bursts 086059-IW2 and 086060-IW2.

Same bbox, same WorldCover classes, same statistic (class mean of linear power, in dB) as the
NISAR series, so the L- vs C-band responses to the recession can be compared date by date.
The two bursts are combined by pooling linear-power sums and pixel counts per class (pixels
in the burst overlap are counted once, from the first burst). OPERA mask: 0 = valid;
shadow/layover/nodata are excluded.

Outputs: processed/reanalysis_v2/phase18_hakaluki_s1_timeseries.csv
"""
from pathlib import Path
from collections import defaultdict
import csv
import re
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling, transform_bounds

ROOT = Path(r"D:\NISAR_Dhaka_Flood")
OPERA = ROOT / "raw_data" / "opera_rtc_s1"
WC_TILE = ROOT / "raw_data" / "worldcover" / "ESA_WorldCover_10m_2020_v100_N24E090_Map.tif"
OUT = ROOT / "processed" / "reanalysis_v2" / "phase18_hakaluki_s1_timeseries.csv"
HAKALUKI_BBOX = (398749.23462869856, 2719084.5651329057, 412351.523585242, 2735786.232745403)
WC_CLASSES = {10: "tree cover", 30: "grassland", 40: "cropland", 90: "wetland", 80: "permanent water"}


def read(path, bbox_crs):
    with rasterio.open(path) as s:
        b = transform_bounds("EPSG:32646", s.crs, *bbox_crs)
        win = from_bounds(*b, transform=s.transform).intersection(
            rasterio.windows.Window(0, 0, s.width, s.height))
        a = s.read(1, window=win).astype("float64")
        return a, s.window_transform(win), s.crs, s.nodata


acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0, 0]))  # date -> cls -> [sumVV, nVV, sumVH, nVH]
sensor = {}
seen = defaultdict(list)  # date -> list of (crs, bounds) already counted
for vv_path in sorted(OPERA.glob("*_VV.tif")):
    m = re.search(r"_(T041-\d+-IW\d)_(2026\d{4})T.*_(S1[A-D])_", vv_path.name)
    burst, date, sat = m.groups()
    sensor[date] = sat
    vh_path = Path(str(vv_path).replace("_VV.tif", "_VH.tif"))
    mk_path = Path(str(vv_path).replace("_VV.tif", "_mask.tif"))
    if not (vh_path.exists() and mk_path.exists()):
        print(f"{date} {burst}: missing VH or mask, skipped")
        continue
    vv, t, crs, nod = read(vv_path, HAKALUKI_BBOX)
    vh, _, _, _ = read(vh_path, HAKALUKI_BBOX)
    mk, _, _, _ = read(mk_path, HAKALUKI_BBOX)
    wc = np.zeros(vv.shape, dtype=np.uint8)
    with rasterio.open(WC_TILE) as s:
        reproject(rasterio.band(s, 1), wc, dst_transform=t, dst_crs=crs, resampling=Resampling.nearest)
    valid = (mk == 0) & (vv > 0) & (vh > 0) & np.isfinite(vv) & np.isfinite(vh)
    # drop pixels already counted from the other burst on this date (overlap)
    ys, xs = np.mgrid[0:vv.shape[0], 0:vv.shape[1]]
    X, Y = rasterio.transform.xy(t, ys.ravel(), xs.ravel())
    X = np.asarray(X).reshape(vv.shape)
    Y = np.asarray(Y).reshape(vv.shape)
    for (c0, (x0, y0, x1, y1)) in seen[date]:
        if c0 == crs:
            valid &= ~((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1))
    xs_v, ys_v = X[valid], Y[valid]
    if xs_v.size:
        seen[date].append((crs, (xs_v.min(), ys_v.min(), xs_v.max(), ys_v.max())))
    for code, name in WC_CLASSES.items():
        sel = valid & (wc == code)
        a = acc[date][name]
        a[0] += vv[sel].sum(); a[1] += int(sel.sum())
        a[2] += vh[sel].sum(); a[3] += int(sel.sum())

rows = []
for date in sorted(acc):
    print(f"\n{date} ({sensor[date]})")
    for name in WC_CLASSES.values():
        s_vv, n, s_vh, _ = acc[date][name]
        if n < 100:
            continue
        vvdb, vhdb = 10 * np.log10(s_vv / n), 10 * np.log10(s_vh / n)
        rows.append(dict(date=date, sensor=sensor[date], cls=name, n=n, vv_db=vvdb, vh_db=vhdb,
                         vv_minus_vh_db=vvdb - vhdb))
        print(f"  {name:16s} VV {vvdb:6.2f}  VH {vhdb:6.2f}  VV-VH {vvdb - vhdb:5.2f}  n={n:,}")

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nWrote {OUT}")
