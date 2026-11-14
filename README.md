# Code: MSc EMDA Diss P2630554

All scripts find their inputs relative to the project folder (two levels above each script),
so the folder can be moved. Run the steps in order; steps 4-7 are chained in `run_all.bat`.

Environment: Python 3.10, with `numpy pandas scipy statsmodels matplotlib rasterio netCDF4 h5py scikit-image`
(ACOLITE for step 3; `eo-tides` for tides; `copernicusmarine` for SST).

| Step | Script | What it does | Main output |
| 1 | `01_download/s2_pipeline.py` (uses `cdse_client.py`) | Search CDSE and download Sentinel-2 L1C, one scene per week window, lowest cloud | `downloaded/`, `manifest.csv` |
| 1b | `01_download/download_atl24_saadiyat.py` | Download ICESat-2 ATL24 bathymetry granules | `E:\atl24_data\` |
| 2 | `02_metadata/build_image_log.py` | Scene time, sun/view geometry, cloud % from .SAFE metadata | `image_log.csv` |
| 3 | `03_acolite/run_acolite_aoi2.py --mode both` | ACOLITE DSF, without and with residual glint correction | `acolite_aoi2_noglint/`, `acolite_aoi2_glint/` |
| 4a | (QGIS) | Exclusion polygons for dredged channels, edited in QGIS | `aoi/exclude_dredged.geojson` |
| 4b | `04_extract/extract_scene_table.py` (stages `masks`, `extract`, `summary`) | Mask hierarchy and one row per scene (indices per class, turbidity, glint, geometry, flags) | `extraction_aoi2_noglint/masks_v5.tif`, `scene_table_aoi2_*.csv` |
| 4c | `04_extract/make_aoi_composite.py` | True-colour median composite of the analysed area (Fig 1 base) | `extraction_aoi2_noglint/composite_truecolour_aoi.tif` |
| 4d | `04_extract/annual_change.py` | 2019 vs 2023 seabed / coastline change (dredging, reclamation) | `extraction_aoi2_noglint/annual/change_2019_2023.tif` |
| 3b | `06_ancillary/cloud_fraction.py` | Cloud fraction per scene over analysed seabed and reference water (L1C cloud mask); >1 % -> `flag_cloud` | `ancillary/cloud_fraction_l1c.csv` |
| 5 | `06_ancillary/tides.py`, `sst.py`, `sst_longterm.py` | EOT20 tide at overpass; OSTIA SST per scene; SST 2015-2025 context | `ancillary/*.csv` |
| 6 | `05_analysis/rq1_seasonal.py` | RQ1: annual harmonic per index and class | `results/rq1_harmonic_aoi2.csv` |
| 6 | `05_analysis/rq2_rq3.py` | RQ2: confound regression (HAC SEs, LMG, VIF); RQ3: SAV minus sand | `results/rq2_rq3_aoi2.csv`, `rq3_veg_minus_sand_aoi2.csv` |
| 6 | `05_analysis/rq4_depth.py` (stages `photons`, `extract`, `analyse`) | RQ4: seasonal cycle by ATL24 depth bin | `results/rq4_depth_aoi2.csv` |
| 7 | `07_figures/make_figures.py` | Figures 2-7 and Tables 1-2 | `figures/` |

