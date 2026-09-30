# NISAR L-band vs Sentinel-1 C-band vegetated-flood-detection code

Processing scripts for:

**Ahmed, F.** "Testing Vegetated-Flood Detection with NISAR and Sentinel-1 in Bangladesh: Null Cases, Acquisition-Mode Discrepancies, and Scene-Pair Quality Assurance." Submitted to the *International Journal of Applied Earth Observation and Geoinformation*.

Solo, unfunded undergraduate research, Department of Urban and Regional Planning (URP), Bangladesh University of Engineering and Technology (BUET).

## What this is, and what it is not

This repository holds the processing and figure-generation scripts used to produce every table and figure in the manuscript, deposited so the analysis is independently checkable and reproducible from the raw inputs. It does **not** host the raw satellite data itself. NISAR Level-2 GCOV and Sentinel-1 SAR data are public and already cited in the manuscript's Data Availability statement (Alaska Satellite Facility DAAC, https://search.asf.alaska.edu); ESA WorldCover 10 m (https://esa-worldcover.org) and NASA POWER (https://power.larc.nasa.gov) are likewise public. Re-hosting multi-gigabyte HDF5 granules and Sentinel-1 SLC archives here would be redundant with those archives and is not attempted.

## Script-to-manuscript map

| Script | Manuscript section(s) | Purpose |
|---|---|---|
| `phase2_preprocessing.py` | Section 3.2, 4.2 | NISAR GCOV HDF5 windowed extraction, Sentinel-1 RTC read, reprojection/co-registration to the common 20 m grid, WorldCover masking, initial absolute-value clustering QA check |
| `phase4_preprocessing.py` | Section 3.3 | Two-experiment redesign: re-extraction and co-registration of the rainfall-screened, genuinely-dry-baseline scene pairs (Experiments 1 and 2) |
| `phase4_differencing.py` | Section 3.4, 4.3, 4.4 (Tables 3, 4) | Class-mean differencing, Demra, both experiments, both sensors; produces the since-superseded initial result and the corrected main result |
| `phase5_native10m_check.py` | Section 3.5, 4.5 (Fig. 6) | Native-10 m robustness re-analysis, Experiment 1 NISAR |
| `phase6_uttarkhan.py` | Section 4.4 (Table 5) | Uttar Khan differencing, Experiment 1 |
| `phase6_diagnose_exp2.py` | Section 3.6, 4.7 (Table 6, Fig. 7) | Diagnosis of the Uttar Khan Experiment 2 acquisition-mode calibration discrepancy: permanent-water shift comparison across mode-matched/mismatched scene pairs |
| `phase8_copol_check.py` | Section 4.4 discussion, Uttar Khan co-pol robustness check | NISAR HHHH / Sentinel-1 VV co-polarization cross-check at Uttar Khan |
| `phase9_replicate_ascending.py` | Section 4.4 footnote (second independent date-pair replication) | Extends the Demra/Uttar Khan record with a further ascending-track date pair |
| `phase10_hakaluki_haor.py` | Section 4.9 (Table 8) | Hakaluki Haor cross-polarized (NISAR HVHV, Sentinel-1 VH) differencing and permanent-water calibration check |
| `phase11_hakaluki_copol.py` | Section 4.10 (Table 9) | Hakaluki Haor co-polarized (NISAR HHHH, Sentinel-1 VV) differencing and calibration check |
| `phase12_kaptai_lake.py` | Section 4.11 (Tables 10, 11) | Kaptai Lake three-channel (NISAR HVHV, Sentinel-1 VH/VV) differencing and calibration check |
| `phase12_kaptai_s1_vv_check.py` | Section 4.11 | Kaptai Lake Sentinel-1 VV co-polarization cross-check |
| `qa_spatial_bootstrap.py` | Section 4.12 (Table 12) | Pixel-count-weighted, non-overlapping spatial block bootstrap (5×5-pixel blocks, 3,000 resamples) producing 95% confidence intervals for all ten permanent-water calibration checks |
| `make_fig1_maps.py` | Fig. 1, Supplementary Fig. S1 | Study-area and locator maps over ESA WorldCover |
| `make_figs_2to7.py` | Figs. 2–7 | WorldCover composition, rainfall time series, delta heatmap, pixel-anomaly fractions, native-resolution robustness, calibration-artifact figure |
| `fix_fig5.py` | Fig. 5 | Layout correction for the pixel-anomaly-fraction figure (annotation/title overlap fix) |
| `make_fig8_qa_spatial.py` | Fig. 8 | Spatial difference maps for the four permanent-water QA diagnostics |

Table 13 (the NASA POWER wind-speed cross-check, Section 4.12) was produced by retrieving the NASA POWER daily-wind API response directly (no local script; the API call and its parameters are documented in the manuscript text and are trivially reproducible from the URL pattern: `https://power.larc.nasa.gov/api/temporal/daily/point?parameters=WS10M,WS10M_MAX&community=RE&longitude=...&latitude=...&start=...&end=...&format=JSON`).

## Environment

Python 3.10, with `rasterio`, `h5py`, `numpy`, `scipy`, `matplotlib`, and `pyproj`/GDAL available (GDAL's `/vsizip/` virtual filesystem is used to read Sentinel-1 RTC bands directly from HyP3's delivered zip archives without extracting them). No `requirements.txt` is pinned yet; if reproducing this exactly matters to a reviewer, open an issue and this will be added, but the scripts do not depend on unusual or version-sensitive APIs from any of these libraries.

## Provenance note

These scripts were run interactively, one processing phase at a time, over several days in September 2026, on a single researcher's personal workstation, not as a packaged end-to-end pipeline with a single entry point. They are deposited here as-is, exactly as run, rather than refactored into a cleaner pipeline after the fact, so that what is published here is what actually produced the manuscript's numbers.
