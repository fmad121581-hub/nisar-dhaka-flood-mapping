import os, json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

_HOME = Path(os.environ["HOME"])
D_ROOT = _HOME / "mnt" / "d" / "NISAR_Dhaka_Flood"
PROC = D_ROOT / "processed"
FIG_DIR = D_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

WATER_CODE = 80

def read_band(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        shape = (src.height, src.width)
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, transform, crs, shape

def water_mask_matching(wc_path, ref_transform, ref_crs, ref_shape):
    with rasterio.open(wc_path) as src:
        wc = src.read(1)
        wc_transform = src.transform
        wc_crs = src.crs
        wc_shape = (src.height, src.width)
    if wc_shape == ref_shape:
        # still verify transform roughly matches; if so, use directly
        return (wc == WATER_CODE)
    # resample to match reference grid (nearest neighbor, categorical)
    dst = np.zeros(ref_shape, dtype=wc.dtype)
    reproject(
        source=wc, destination=dst,
        src_transform=wc_transform, src_crs=wc_crs,
        dst_transform=ref_transform, dst_crs=ref_crs,
        resampling=Resampling.nearest,
    )
    return (dst == WATER_CODE)

def delta_over_water(pre_path, during_path, wc_path, to_db=False):
    pre, tr, crs, shp = read_band(pre_path)
    during, tr2, crs2, shp2 = read_band(during_path)
    assert shp == shp2, f"shape mismatch {shp} vs {shp2} ({pre_path} vs {during_path})"
    if to_db:
        with np.errstate(divide='ignore'):
            pre = 10*np.log10(pre)
            during = 10*np.log10(during)
    delta = during - pre
    wmask = water_mask_matching(wc_path, tr, crs, shp)
    valid = wmask & np.isfinite(delta)
    return delta, valid, tr

def block_bootstrap_ci(delta, valid_mask, block=5, n_boot=3000, seed=42):
    """Non-overlapping spatial block bootstrap, weighted by block pixel count
    so the bootstrap distribution is centered on the true pixel-weighted mean
    (rather than an unweighted mean-of-block-means, which is biased when
    blocks along an irregular shoreline carry very unequal pixel counts)."""
    rng = np.random.default_rng(seed)
    H, W = delta.shape
    block_means, block_counts = [], []
    for i in range(0, H, block):
        for j in range(0, W, block):
            sub_valid = valid_mask[i:i+block, j:j+block]
            n = int(sub_valid.sum())
            if n == 0:
                continue
            sub_delta = delta[i:i+block, j:j+block]
            block_means.append(np.nanmean(sub_delta[sub_valid]))
            block_counts.append(n)
    block_means = np.array(block_means)
    block_counts = np.array(block_counts, dtype=np.float64)
    n_blocks = len(block_means)
    if n_blocks == 0:
        return None
    overall_mean = float(np.sum(block_means * block_counts) / np.sum(block_counts))
    idx = np.arange(n_blocks)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(idx, size=n_blocks, replace=True)
        w = block_counts[sample]
        boot_means[b] = np.sum(block_means[sample] * w) / np.sum(w)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "n_water_px": int(valid_mask.sum()),
        "n_blocks": n_blocks,
        "block_px": block,
        "mean": overall_mean,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
    }

results = {}

# --- Demra: 2 Jul -> 14 Jul NISAR HVHV (mode-mismatched pair, same as reported in Table 6) ---
DEM = PROC / "coregistered"
results["Demra_NISAR_2Jul_14Jul"] = block_bootstrap_ci(
    *delta_over_water(DEM/"nisar_july2_desc_HVHV_coreg.tif", DEM/"nisar_july14_desc_HVHV_coreg.tif", PROC/"worldcover_clipped.tif")[:2]
)

# --- Uttar Khan: mode-mismatched (2 Jul -> 14 Jul) and mode-matched (30 Jun -> 12 Jul) ---
UK = PROC / "uttarkhan_case" / "coregistered"
uk_wc = UK / "worldcover_coreg.tif"
delta_uk_mismatch, valid_uk_mismatch, _ = delta_over_water(UK/"nisar_july2_desc_coreg.tif", UK/"nisar_july14_desc_coreg.tif", uk_wc)
results["UttarKhan_NISAR_mismatched_2Jul_14Jul"] = block_bootstrap_ci(delta_uk_mismatch, valid_uk_mismatch)

delta_uk_match, valid_uk_match, _ = delta_over_water(UK/"nisar_june30_asc_coreg.tif", UK/"nisar_july12_asc_coreg.tif", uk_wc)
results["UttarKhan_NISAR_matched_30Jun_12Jul"] = block_bootstrap_ci(delta_uk_match, valid_uk_match)

# --- Hakaluki Haor: cross-pol (NISAR HVHV, S1 VH) and co-pol (NISAR HHHH, S1 VV) ---
HH_XP = PROC / "hakaluki_haor_case" / "coregistered"
d, v, _ = delta_over_water(HH_XP/"nisar_pre_coreg.tif", HH_XP/"nisar_during_coreg.tif", HH_XP/"worldcover_coreg.tif")
results["Hakaluki_NISAR_HVHV"] = block_bootstrap_ci(d, v)
d, v, _ = delta_over_water(HH_XP/"s1_pre_coreg.tif", HH_XP/"s1_during_coreg.tif", HH_XP/"worldcover_coreg.tif")
results["Hakaluki_S1_VH"] = block_bootstrap_ci(d, v)

HH_CP = PROC / "hakaluki_haor_case_copol" / "coregistered"
d, v, _ = delta_over_water(HH_CP/"nisar_pre_coreg.tif", HH_CP/"nisar_during_coreg.tif", HH_CP/"worldcover_coreg.tif")
results["Hakaluki_NISAR_HHHH"] = block_bootstrap_ci(d, v)
d, v, hh_vv_transform = delta_over_water(HH_CP/"s1_pre_coreg.tif", HH_CP/"s1_during_coreg.tif", HH_CP/"worldcover_coreg.tif")
results["Hakaluki_S1_VV"] = block_bootstrap_ci(d, v)
hh_vv_delta, hh_vv_valid = d, v

# --- Kaptai Lake: NISAR HVHV, S1 VH, S1 VV ---
KL = PROC / "kaptai_lake_case" / "coregistered"
d, v, _ = delta_over_water(KL/"nisar_pre_coreg.tif", KL/"nisar_during_coreg.tif", KL/"worldcover_coreg.tif")
results["Kaptai_NISAR_HVHV"] = block_bootstrap_ci(d, v)
d, v, _ = delta_over_water(KL/"s1_pre_coreg.tif", KL/"s1_during_coreg.tif", KL/"worldcover_coreg.tif")
results["Kaptai_S1_VH"] = block_bootstrap_ci(d, v)
d, v, kl_vv_transform = delta_over_water(KL/"s1vv_pre_coreg.tif", KL/"s1vv_during_coreg.tif", KL/"worldcover_coreg.tif")
results["Kaptai_S1_VV"] = block_bootstrap_ci(d, v)
kl_vv_delta, kl_vv_valid = d, v

with open(D_ROOT / "processed" / "qa_bootstrap_results.json", "w") as f:
    json.dump(results, f, indent=2)

for k, r in results.items():
    if r is None:
        print(k, "NO WATER PIXELS")
    else:
        print(f"{k}: mean={r['mean']:.3f} dB, 95% CI=[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}], n_water_px={r['n_water_px']}, n_blocks={r['n_blocks']}")

# stash arrays for the spatial-map figure step
np.savez(D_ROOT / "processed" / "qa_spatial_arrays.npz",
         uk_mismatch_delta=delta_uk_mismatch, uk_mismatch_valid=valid_uk_mismatch,
         uk_match_delta=delta_uk_match, uk_match_valid=valid_uk_match,
         hh_vv_delta=hh_vv_delta, hh_vv_valid=hh_vv_valid,
         kl_vv_delta=kl_vv_delta, kl_vv_valid=kl_vv_valid)
print("Saved qa_bootstrap_results.json and qa_spatial_arrays.npz")
