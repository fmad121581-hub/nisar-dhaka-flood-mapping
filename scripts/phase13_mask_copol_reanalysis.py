"""
Phase 13 -- Re-analysis with the GCOV valid-sample mask applied, co-pol (HHHH) as primary.

Why this exists (2026-10-23, after the NHESS desk rejection):
  1. None of phases 2-12 applied GCOV's own `mask` layer. Its definition (read from the
     HDF5 attribute): "A GCOV sample is only considered valid if it is generated from
     fully-focused radar samples ... the corresponding mask pixel will contain the value 0.
     GCOV pixels outside of the radar acquisition extent are filled with the value 255."
     The 14 Jul QPDH granule is a partial frame (isFullFrame=False) and ~97% of the Uttar Khan
     bbox is mask=0 in it -- the "8 dB acquisition-mode artifact" of Section 4.7 needs to be
     re-examined against this.
  2. Double-bounce enhancement is classically strongest in co-pol (HH at L-band); the
     manuscript used cross-pol (HVHV) as primary. This run reports HHHH and HVHV side by side.

Reads the raw GCOV HDF5 directly (windowed), so it does not depend on earlier phases' rasters.
WorldCover 2020 is warped (nearest) onto each NISAR grid.

Outputs: processed/reanalysis_v2/phase13_results.csv and printed tables.
Needs: h5py rasterio numpy
"""
from pathlib import Path
import csv
import numpy as np
import h5py
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin

ROOT = Path(r"D:\NISAR_Dhaka_Flood")
GCOV = ROOT / "raw_data" / "nisar_gcov"
WC_TILE = ROOT / "raw_data" / "worldcover" / "ESA_WorldCover_10m_2020_v100_N21E090_Map.tif"
OUT_DIR = ROOT / "processed" / "reanalysis_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CRS = "EPSG:32646"
SITES = {
    "Demra": (240673.90374, 2621657.94409, 246404.89660, 2630128.18446),
    "UttarKhan": (236582.74858659075, 2639492.080730549, 242585.99816525163, 2645681.772454119),
}
EXPERIMENTS = {
    "Exp1_asc": ("*20260630T*.h5", "*20260712T*.h5"),
    "Exp2_desc": ("*20260702T*.h5", "*20260714T*.h5"),
}
WC_CLASSES = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland",
              50: "built-up", 60: "bare/sparse", 80: "permanent water", 90: "wetland"}
VEG = [10, 20, 30, 40, 90]
GRID = "science/LSAR/GCOV/grids/frequencyA/"


def granule(pattern):
    hits = [p for p in GCOV.glob(pattern) if p.suffix == ".h5"]
    assert len(hits) == 1, (pattern, hits)
    return hits[0]


def read_window(path, bbox):
    """Return dict of HH/HV (linear), mask, transform for the bbox window."""
    x0, y0, x1, y1 = bbox
    with h5py.File(path, "r") as f:
        x = f[GRID + "xCoordinates"][:]
        y = f[GRID + "yCoordinates"][:]
        xi = np.where((x >= x0) & (x <= x1))[0]
        yi = np.where((y >= y0) & (y <= y1))[0]
        sl = (slice(yi.min(), yi.max() + 1), slice(xi.min(), xi.max() + 1))
        dx = x[1] - x[0]
        dy = y[1] - y[0]  # negative
        out = {
            "HH": f[GRID + "HHHH"][sl],
            "HV": f[GRID + "HVHV"][sl],
            "mask": f[GRID + "mask"][sl],
            "full_frame": f["science/LSAR/identification/isFullFrame"][()].decode(),
            # pixel-centre coordinates -> upper-left corner
            "transform": from_origin(x[xi.min()] - dx / 2, y[yi.min()] - dy / 2, dx, -dy),
            "spacing": dx,
        }
    return out


def worldcover_on(transform, shape):
    dst = np.zeros(shape, dtype=np.uint8)
    with rasterio.open(WC_TILE) as src:
        reproject(source=rasterio.band(src, 1), destination=dst,
                  dst_transform=transform, dst_crs=TARGET_CRS,
                  resampling=Resampling.nearest)
    return dst


def to_db(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a > 0, 10 * np.log10(a), np.nan)


rows = []
for site, bbox in SITES.items():
    for exp, (pre_pat, dur_pat) in EXPERIMENTS.items():
        pre = read_window(granule(pre_pat), bbox)
        dur = read_window(granule(dur_pat), bbox)
        assert pre["HH"].shape == dur["HH"].shape, "pre/during grids differ"
        wc = worldcover_on(pre["transform"], pre["HH"].shape)
        valid_pre = (pre["mask"] > 0) & (pre["mask"] < 255)
        valid_dur = (dur["mask"] > 0) & (dur["mask"] < 255)
        valid = valid_pre & valid_dur
        n_bbox = valid.size
        print(f"\n== {site} {exp} ({pre['spacing']:.0f} m) full_frame pre/dur = "
              f"{pre['full_frame']}/{dur['full_frame']}; valid {valid.sum():,}/{n_bbox:,} "
              f"({100 * valid.sum() / n_bbox:.1f}%)")
        for pol in ("HH", "HV"):
            d = to_db(dur[pol]) - to_db(pre[pol])
            for cls in VEG + [50, 80]:
                for label, keep in (("masked", valid), ("unmasked", np.ones_like(valid))):
                    sel = (wc == cls) & keep & np.isfinite(d)
                    n = int(sel.sum())
                    if n < 30:
                        mean = std = up = dn = np.nan
                    else:
                        v = d[sel]
                        mean, std = float(v.mean()), float(v.std())
                        up, dn = float((v > 3).mean() * 100), float((v < -3).mean() * 100)
                    rows.append(dict(site=site, experiment=exp, pol=pol, cls=WC_CLASSES[cls],
                                     mask=label, n=n, mean_db=mean, std_db=std,
                                     pct_gt_plus3=up, pct_lt_minus3=dn))
            print(f"  {pol}: " + "  ".join(
                f"{r['cls']}: {r['mean_db']:+.2f} (n={r['n']})"
                for r in rows if r["site"] == site and r["experiment"] == exp
                and r["pol"] == pol and r["mask"] == "masked"))
            print(f"  {pol} unmasked water: " + "".join(
                f"{r['mean_db']:+.2f} (n={r['n']})" for r in rows
                if r["site"] == site and r["experiment"] == exp and r["pol"] == pol
                and r["mask"] == "unmasked" and r["cls"] == "permanent water"))

with open(OUT_DIR / "phase13_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nWrote {OUT_DIR / 'phase13_results.csv'}")
