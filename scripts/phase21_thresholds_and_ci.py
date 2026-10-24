"""
Phase 21 -- Sensor-specific water thresholds and block-bootstrap confidence intervals.

Replaces the single -15 dB cut of phases 19-20 with ONE Otsu threshold per site and sensor,
pooled over all dates (NISAR HH, Sentinel-1 VV). Per-scene Otsu breaks when the box is almost
all water (unimodal histogram); pooling keeps it sensor-specific but fixed in time. Reports:
  - class mean of linear power in dB, with a 95 % spatial block-bootstrap CI
    (5 x 5-pixel blocks, 2,000 resamples, block sums and counts resampled together);
  - share of pixels below that scene's Otsu threshold ("water-like").

Sites: Hakaluki (NISAR 141/014, S1 OPERA T041 bursts 086059-086060) and Kaptai (NISAR
069/013, S1 OPERA T041 bursts 086047-086050, restricted to the NISAR valid footprint).
22 Aug 2026 NISAR excluded (corrupt granule).

Outputs: processed/reanalysis_v2/phase21_thresholds_ci.csv
"""
from pathlib import Path
import csv
import re
from collections import defaultdict
import numpy as np
import h5py
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

ROOT = Path(r"D:\NISAR_Dhaka_Flood")
GCOV = ROOT / "raw_data" / "nisar_gcov"
OPERA = ROOT / "raw_data" / "opera_rtc_s1"
DOWNLOADS = Path(r"C:\Users\user\Downloads")  # the 23 Jun Hakaluki granule lives here
OUT = ROOT / "processed" / "reanalysis_v2" / "phase21_thresholds_ci.csv"
CRS = "EPSG:32646"
GRID = "science/LSAR/GCOV/grids/frequencyA/"
WC = {10: "tree cover", 30: "grassland", 40: "cropland", 90: "wetland", 80: "permanent water"}
EXCLUDE = {"20260822"}
RNG = np.random.default_rng(42)
BLOCK, NBOOT = 5, 2000

SITES = {
    "Hakaluki": dict(
        box=(398749.23462869856, 2719084.5651329057, 412351.523585242, 2735786.232745403),
        wc=ROOT / "raw_data" / "worldcover" / "ESA_WorldCover_10m_2020_v100_N24E090_Map.tif",
        nisar="NISAR_L2_PR_GCOV_*_141_A_014_4005_DHDH_*_001.h5",
        bursts=("086059", "086060"), footprint=False),
    "Kaptai": dict(
        box=(402869.90916713024, 2485087.8977870205, 433476.6124612646, 2549171.3867336917),
        wc=ROOT / "raw_data" / "worldcover" / "ESA_WorldCover_10m_2020_v100_N21E090_Map.tif",
        nisar="NISAR_L2_PR_GCOV_*_069_A_013_4005_DHDH_*_001.h5",
        bursts=("086047", "086048", "086049", "086050"), footprint=True),
}


def otsu(db_values, bins=256):
    v = db_values[np.isfinite(db_values)]
    v = v[(v > -35) & (v < 5)]
    hist, edges = np.histogram(v, bins=bins)
    mids = (edges[:-1] + edges[1:]) / 2
    w = hist.cumsum()
    m = (hist * mids).cumsum()
    wt, mt = w[-1], m[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w / wt - m) ** 2 / (w * (wt - w))
    return float(mids[np.nanargmax(between)])


def block_ci(lin, sel):
    """95 % CI of 10*log10(mean linear power) by resampling 5x5 blocks."""
    h, w = sel.shape
    hb, wb = h // BLOCK, w // BLOCK
    s = np.where(sel, lin, 0.0)[:hb * BLOCK, :wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK).sum((1, 3)).ravel()
    n = sel[:hb * BLOCK, :wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK).sum((1, 3)).ravel()
    keep = n > 0
    s, n = s[keep], n[keep]
    if n.size < 20:
        return np.nan, np.nan
    idx = RNG.integers(0, n.size, size=(NBOOT, n.size))
    means = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(10 * np.log10(means), [2.5, 97.5])
    return float(lo), float(hi)


EDGES = np.arange(-35.0, 5.0001, 0.05)
POOLED = defaultdict(lambda: np.zeros(EDGES.size - 1))   # (site, sensor) -> histogram of all valid px
CLASS_HIST = {}                                          # row index -> class histogram


def otsu_hist(hist):
    mids = (EDGES[:-1] + EDGES[1:]) / 2
    w, m = hist.cumsum(), (hist * mids).cumsum()
    denom = w * (w[-1] - w)
    ok = denom > 0  # empty leading/trailing bins would give 0/0 or x/0
    between = np.full(hist.shape, -np.inf)
    between[ok] = (m[-1] * w[ok] / w[-1] - m[ok]) ** 2 / denom[ok]
    return float(mids[np.argmax(between)])


def summarize(site, sensor, date, co_lin, cross_lin, valid, wc, rows):
    co_db = np.where(valid, 10 * np.log10(np.where(co_lin > 0, co_lin, np.nan)), np.nan)
    POOLED[(site, sensor)] += np.histogram(co_db[valid], bins=EDGES)[0]
    for code, name in WC.items():
        sel = valid & (wc == code)
        if sel.sum() < 200:
            continue
        mean = 10 * np.log10(co_lin[sel].mean())
        lo, hi = block_ci(co_lin, sel)
        ratio = mean - 10 * np.log10(cross_lin[sel].mean())
        CLASS_HIST[len(rows)] = np.histogram(co_db[sel], bins=EDGES)[0]
        rows.append(dict(site=site, sensor=sensor, date=date, cls=name, n=int(sel.sum()),
                         co_db=round(mean, 3), co_ci_lo=round(lo, 3), co_ci_hi=round(hi, 3),
                         co_minus_cross_db=round(ratio, 3)))
    print(f"{site:8s} {sensor:5s} {date}  " + "  ".join(
        f"{r['cls'][:5]} {r['co_db']:.2f} [{r['co_ci_lo']:.2f},{r['co_ci_hi']:.2f}]"
        for r in rows if r["site"] == site and r["sensor"] == sensor and r["date"] == date))


rows = []
for site, cfg in SITES.items():
    x0, y0, x1, y1 = cfg["box"]
    # ---- NISAR on its own grid
    nis_valid_30 = None
    seen_dates = set()
    for p in sorted(GCOV.glob(cfg["nisar"])) + sorted(DOWNLOADS.glob(cfg["nisar"])):
        date = re.search(r"_(2026\d{4})T", p.name).group(1)
        if date in EXCLUDE or date in seen_dates:
            continue
        try:
            f = h5py.File(p, "r")
        except OSError:  # truncated copy
            continue
        seen_dates.add(date)
        with f:
            x, y = f[GRID + "xCoordinates"][:], f[GRID + "yCoordinates"][:]
            xi = np.where((x >= x0) & (x <= x1))[0]
            yi = np.where((y >= y0) & (y <= y1))[0]
            sl = (slice(yi.min(), yi.max() + 1), slice(xi.min(), xi.max() + 1))
            hh, hv, m = f[GRID + "HHHH"][sl], f[GRID + "HVHV"][sl], f[GRID + "mask"][sl]
            t = from_origin(x[xi.min()] - (x[1] - x[0]) / 2, y[yi.min()] - (y[1] - y[0]) / 2,
                            x[1] - x[0], y[0] - y[1])
        valid = (m > 0) & (m < 255) & (hh > 0) & (hv > 0)
        wc = np.zeros(hh.shape, np.uint8)
        with rasterio.open(cfg["wc"]) as s:
            reproject(rasterio.band(s, 1), wc, dst_transform=t, dst_crs=CRS, resampling=Resampling.nearest)
        summarize(site, "NISAR", date, hh.astype("float64"), hv.astype("float64"), valid, wc, rows)
        if nis_valid_30 is None:
            nis_valid_30 = (valid, t)
    # ---- Sentinel-1 OPERA mosaicked on a 30 m grid
    shape = (int((y1 - y0) / 30), int((x1 - x0) / 30))
    T = from_origin(x0, y1, 30, 30)
    wc30 = np.zeros(shape, np.uint8)
    with rasterio.open(cfg["wc"]) as s:
        reproject(rasterio.band(s, 1), wc30, dst_transform=T, dst_crs=CRS, resampling=Resampling.mode)
    fp = np.ones(shape, bool)
    if cfg["footprint"]:
        v, t = nis_valid_30
        tmp = np.zeros(shape, np.uint8)
        reproject(v.astype(np.uint8), tmp, src_transform=t, src_crs=CRS, dst_transform=T, dst_crs=CRS,
                  resampling=Resampling.mode)
        fp = tmp.astype(bool)
    by_date = defaultdict(list)
    for vv in sorted(OPERA.glob("OPERA_L2_RTC-S1_T041-*_VV.tif")):
        if any(f"T041-{b}-" in vv.name for b in cfg["bursts"]):
            by_date[re.search(r"_(2026\d{4})T", vv.name).group(1)].append(vv)

    def grid(path, rs):
        d = np.full(shape, np.nan, "float32")
        with rasterio.open(path) as s:
            reproject(rasterio.band(s, 1), d, dst_transform=T, dst_crs=CRS, resampling=rs,
                      src_nodata=s.nodata, dst_nodata=np.nan)
        return d

    for date, paths in sorted(by_date.items()):
        VV = np.full(shape, np.nan, "float32")
        VH = np.full(shape, np.nan, "float32")
        for vv in paths:
            a = grid(vv, Resampling.bilinear)
            b = grid(Path(str(vv).replace("_VV.tif", "_VH.tif")), Resampling.bilinear)
            mk = grid(Path(str(vv).replace("_VV.tif", "_mask.tif")), Resampling.nearest)
            ok = (mk == 0) & (a > 0) & (b > 0) & np.isnan(VV)
            VV[ok], VH[ok] = a[ok], b[ok]
        valid = np.isfinite(VV) & np.isfinite(VH) & fp
        if valid.mean() < 0.05:
            continue
        summarize(site, "S1", date, np.nan_to_num(VV).astype("float64"),
                  np.nan_to_num(VH).astype("float64"), valid, wc30, rows)

THR = {k: otsu_hist(h) for k, h in POOLED.items()}
print("\nPooled Otsu thresholds (all dates per site and sensor):",
      {f"{a}/{b}": round(v, 2) for (a, b), v in THR.items()})
mids = (EDGES[:-1] + EDGES[1:]) / 2
for i, r in enumerate(rows):
    thr = THR[(r["site"], r["sensor"])]
    h = CLASS_HIST[i]
    r["threshold_db"] = round(thr, 2)
    r["pct_water_like"] = round(float(h[mids < thr].sum() / h.sum() * 100), 2)
for site in SITES:
    for sensor in ("NISAR", "S1"):
        sub = [r for r in rows if r["site"] == site and r["sensor"] == sensor]
        print(f"\n{site} {sensor}: % below {THR[(site, sensor)]:.2f} dB")
        for d in sorted({r["date"] for r in sub}):
            print(f"  {d}  " + "  ".join(f"{r['cls'][:5]} {r['pct_water_like']:5.1f}" for r in sub if r["date"] == d))

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"Wrote {OUT}")
