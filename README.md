# NISAR L-band vs Sentinel-1 C-band: monsoon flooding in Bangladesh (v2)

Processing scripts for:

**Ahmed, F.** "Early NISAR L-band observations of monsoon flooding in Bangladesh, compared with Sentinel-1 against independent water-level records." Manuscript in preparation.

Solo, unfunded undergraduate research, Department of Urban and Regional Planning (URP), BUET.

## What changed from v1

v1 accompanied an earlier manuscript ("Testing Vegetated-Flood Detection with NISAR and Sentinel-1 in Bangladesh…", not published). Its scripts are kept unchanged in `scripts/v1/`.

The v2 re-analysis found that v1's headline result was wrong. v1 reported an ~8 dB "acquisition-mode" calibration shift over permanent water at Uttar Khan, 2 → 14 July 2026. That shift came from invalid samples in a partial-frame granule (`isFullFrame = False`; GCOV `mask` = 0 over 96.7 % of the ward). The v1 scripts never applied the valid-sample mask.

v2 makes the following changes:
- Applies the GCOV valid-sample mask.
- Uses co-polarization (HH, VV) as the primary channel.
- Uses stable land references instead of permanent water.
- Adds a Hakaluki Haor time series (NISAR track 141 and OPERA RTC-S1, June–October 2026), read against river stage.
- Adds a Kaptai Lake reservoir-rise case.

## Script map (v2)

| Script | Purpose |
|---|---|
| `phase13_mask_copol_reanalysis.py` | Dhaka, NISAR: masked, HH and HV, class means and changes (the descending pair is computed but excluded from the paper by granule screening) |
| `phase14_s1_copol_reanalysis.py` | Dhaka, Sentinel-1 HyP3 RTC: VV and VH, layover/shadow mask, file integrity check |
| `phase15_builtup_elevation_test.py` | Central Dhaka: built-up co-pol change by Copernicus DSM height band |
| `phase16_hakaluki_timeseries.py` | Hakaluki NISAR class means, every track-141 date |
| `phase17_hakaluki_lowland_vs_upland.py` | Hakaluki basin vs hill tree cover, every date |
| `phase18_hakaluki_s1_opera_timeseries.py` | Hakaluki OPERA RTC-S1 class means, every date |
| `phase19_kaptai_reservoir_rise.py` | Kaptai NISAR (track 069 frame 013). Its Sentinel-1 block reads v1 rasters that do not align with WorldCover and is superseded by phase 20. |
| `phase20_kaptai_s1_opera.py` | Kaptai OPERA RTC-S1, mosaicked and restricted to the NISAR footprint |
| `phase21_thresholds_and_ci.py` | Pooled per-site, per-sensor Otsu thresholds, water-like shares and block-bootstrap CIs (Hakaluki, Kaptai) |
| `phase22_stage_sensitivity.py` | Spearman ρ between Fenchuganj stage and Hakaluki cropland backscatter under all 96 combinations of alternative stage readings (Table 1 and Sect. 2.1) |
| `make_fig_v2_study_area.py` | Fig. 1 |
| `make_fig_v2_qa.py` | Fig. 2 |
| `make_fig_hakaluki_l_vs_c.py` | Fig. 3 and Table 2 metrics |
| `make_fig_v2_kaptai_dhaka.py` | Figs. 4 and 5 |
| `make_supplement_tables.py` | Tables S1–S5 (CSV copies in `tables/`) |
| `build_manuscript_v2_docx.py`, `build_supplement_v2_docx.py` | Word files from the Markdown source (the manuscript text itself is not included here) |

The scripts assume the folder layout `D:\NISAR_Dhaka_Flood\{raw_data,processed,figures}`. Edit `ROOT` at the top of each script to run them elsewhere.

## Inputs (not redistributed)

Study period: SAR acquisitions from 19 June to 12 October 2026 (Hakaluki), 26 June to 1 August 2026 (Kaptai) and 30 June to 13 July 2026 (Dhaka).


- NISAR L2 GCOV provisional, https://doi.org/10.5067/NIL2GCOV-P1
- OPERA RTC-S1, https://doi.org/10.5067/SNWG/OPERA_L2_RTC-S1_V1
- Sentinel-1 IW SLC processed with ASF HyP3
- ESA WorldCover 2020, https://doi.org/10.5281/zenodo.5571936
- Copernicus GLO-30, https://doi.org/10.5270/ESA-c5d3d65 (read remotely as COG)
- geoBoundaries BGD ADM0 and ADM2

Granule identifiers are listed in `tables/tableS1_granules.csv`. River-stage and lake-level records and their sources are in `tables/tableS2_stage_and_levels.csv`.

Hard-coded inputs: the Kaptai Lake levels (in `make_fig_v2_kaptai_dhaka.py` and `make_supplement_tables.py`) and the alternative Fenchuganj readings (in `phase22_stage_sensitivity.py`) are typed in from the sources listed in Table S2; `phase22` writes them to `tables/phase22_stage_alternatives.csv`. Everything else is read from the satellite products or from the CSV outputs of earlier phases.

## Environment

Python 3.13. Pinned versions are in `requirements.txt`.
