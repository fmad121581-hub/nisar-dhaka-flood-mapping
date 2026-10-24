"""
Phase 14 -- Sentinel-1 counterpart of phase 13: VV (co-pol) as primary, VH alongside,
HyP3 layover/shadow map applied, built-up class reported as a stable reference.

Pairs (same as the manuscript):
  Exp1 (ascending, ~12:04 UTC):  2026-07-01 -> 2026-07-13
  Exp2 (descending, ~23:55 UTC): 2026-07-03 -> 2026-07-15
Also integrity-checks every 2026 S1 RTC GeoTIFF (several have stray .partial siblings).

Outputs: processed/reanalysis_v2/phase14_s1_results.csv
"""
from pathlib import Path
import csv
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling, transform_bounds

ROOT = Path(r"D:\NISAR_Dhaka_Flood")
S1 = ROOT / "processed" / "s1_rtc"
WC_TILE = ROOT / "raw_data" / "worldcover" / "ESA_WorldCover_10m_2020_v100_N21E090_Map.tif"
OUT_DIR = ROOT / "processed" / "reanalysis_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SITES = {
    "Demra": (240673.90374, 2621657.94409, 246404.89660, 2630128.18446),
    "UttarKhan": (236582.74858659075, 2639492.080730549, 242585.99816525163, 2645681.772454119),
}
EXPERIMENTS = {
    "Exp1_asc": ("20260701T120341", "20260713T120342"),
    "Exp2_desc": ("20260703T235521", "20260715T235521"),
}
WC_CLASSES = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland",
              50: "built-up", 60: "bare/sparse", 80: "permanent water", 90: "wetland"}
REPORT = [10, 20, 30, 40, 90, 50, 80]


def band_path(stamp, pol):
    hits = list(S1.glob(f"*{stamp}*/*_{pol}.tif"))
    assert len(hits) == 1, (stamp, pol, hits)
    return hits[0]


def integrity():
    print("== Integrity check (full read of every 2026 RTC band)")
    for p in sorted(S1.glob("S1D_IW_2026*/*_V?.tif")):
        try:
            with rasterio.open(p) as src:
                a = src.read(1, masked=True)
                frac = float(np.ma.getmaskarray(a).mean())
                print(f"  OK   {p.name}  {src.width}x{src.height}  masked {frac:.1%}")
        except Exception as e:  # truncated GeoTIFFs fail on full read
            print(f"  FAIL {p.name}: {e}")


def read_site(path, bbox):
    with rasterio.open(path) as src:
        # descending scenes over Dhaka are delivered in UTM 45N, ascending in 46N
        b = transform_bounds("EPSG:32646", src.crs, *bbox)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        a = src.read(1, window=win).astype("float64")
        nod = src.nodata
        tr = src.window_transform(win)
        crs = src.crs
    if nod is not None:
        a[a == nod] = np.nan
    a[a <= 0] = np.nan
    return 10 * np.log10(a), tr, crs


def warp_to(src_path, transform, crs, shape):
    dst = np.zeros(shape, dtype=np.uint8)
    with rasterio.open(src_path) as src:
        reproject(rasterio.band(src, 1), dst, dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.nearest)
    return dst


integrity()  # note: phases 10-12 read the 26 Jun scene from its .zip, not this extracted copy
rows = []
for site, bbox in SITES.items():
    for exp, (pre_s, dur_s) in EXPERIMENTS.items():
        ls_hits = list(S1.glob(f"*{dur_s}*/*_ls_map.tif"))
        for pol in ("VV", "VH"):
            pre, tr, crs = read_site(band_path(pre_s, pol), bbox)
            dur, tr2, _ = read_site(band_path(dur_s, pol), bbox)
            assert pre.shape == dur.shape and tr == tr2, "grids differ"
            wc = warp_to(WC_TILE, tr, crs, pre.shape)
            ok = np.ones(pre.shape, bool)
            if ls_hits:  # HyP3 ls_map: 1 = normal, 0 = nodata, >1 = layover/shadow flags
                ok &= warp_to(ls_hits[0], tr, crs, pre.shape) == 1
            d = dur - pre
            ref = None
            line = []
            for cls in REPORT:
                sel = (wc == cls) & ok & np.isfinite(d)
                n = int(sel.sum())
                v = d[sel] if n >= 30 else np.array([np.nan])
                mean = float(np.nanmean(v))
                if cls == 50:
                    ref = mean
                rows.append(dict(site=site, experiment=exp, pol=pol, cls=WC_CLASSES[cls], n=n,
                                 mean_db=mean, std_db=float(np.nanstd(v)),
                                 pct_gt_plus3=float((v > 3).mean() * 100),
                                 pct_lt_minus3=float((v < -3).mean() * 100)))
                line.append((WC_CLASSES[cls], mean, n))
            print(f"\n{site} {exp} {pol} (ls_map {'applied' if ls_hits else 'n/a'})")
            print("  absolute: " + "  ".join(f"{c} {m:+.2f}" for c, m, n in line))
            print("  rel. built-up: " + "  ".join(f"{c} {m - ref:+.2f}" for c, m, n in line if c != "built-up"))

with open(OUT_DIR / "phase14_s1_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nWrote {OUT_DIR / 'phase14_s1_results.csv'}")
