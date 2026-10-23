# ICAIDM 2026 experimental reproducibility package

## Paper Information

**Internal Discrimination Does Not Predict Ranking Utility: Negative-Sample Construction for Budget-Constrained Spatial Prioritization** — Lu Wang and Yuzhang Qin; ICAIDM 2026 revision, v4. Acceptance status, venue metadata and DOI are not asserted.

## Overview

This package extracts the actual eight-pool Random Forest experiments and their saved sample/split IDs. It separates strategy-conditioned internal ROC-AUC from strict-area event ranking. It includes final v4 results and the paired event/spatial bootstrap implementations. Original algorithms and historical results are not modified.

## Repository Structure

- `src/skills/`: original sampling/modeling modules, byte-identical copies.
- `scripts/`: portable, guarded entry points and statistics scripts.
- `configs/<pool>_seed<seed>/`: adapted actual sampling/model configurations.
- `manifests/splits/`: saved run-local IDs, labels and record order, without coordinates/features.
- `manifests/runs.json`: paper experiment → original relative run ID → portable configuration mapping, recorded metrics and versions.
- `results/`: copied, path-filtered aggregate tables.
- `docs/`: provenance, consistency, validation, license/data boundaries.
- `data/`, `outputs/`: user-supplied inputs and new private outputs, ignored by Git.

## Environment Setup

Python 3.12 was used. For the recorded training environment:

```sh
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment using the command appropriate to your operating system before installing/running. The original statistics runs recorded NumPy 2.2.2, while training metadata records NumPy 2.0.2. Use a separate environment with `requirements-statistics.txt` for matching the statistics version. Do not silently upgrade training dependencies. SciPy, pyproj and pyogrio remain unpinned because their exact versions are not consistently recorded for all runs; authors should resolve this before claiming a fully locked environment. Per-run versions are in `manifests/runs.json`.

## Data Requirements

Private data are intentionally absent. Configure the explicit relative paths in the YAML files; `project_root: ../..` resolves to this repository from each per-run configuration directory. No code relies on the original workstation root.

Required inputs include aligned single-band factor rasters in `data/factors/`, boundary, water polygons, historical inventory and its existing 25 m buffer, the original positive representative points, and the recorded negative sample files. Training inputs must retain the original `sample_id` or positive/negative ID fields and original point order. NS3-family sampling additionally needs the matching seed's NS1 RF raster. Optional comparison sample inputs in original configurations are needed for their original QC; they do not change the sampling rule.

For event evaluation, supply the common valid-domain raster, the original event representative-point table and the final prediction fields. `configs/evaluation.json` lists the path schema and seed-42 example; add the other seed/pool prediction entries explicitly before the bootstrap, which requires NS3/H100/Q60 for all three seeds. Event tables need `row,col,common_valid`; spatial bootstrap additionally needs `event_id,x,y`. `x,y` are projected EPSG:32649 coordinates, not longitude/latitude.

This package does **not** include imagery, shapefiles, GeoPackages, factor values, point coordinates, event-level differences/ranks, block maps, models or prediction rasters. Obtain them under their owners' terms. See DATA_AVAILABILITY.md.

## Environmental Factors

Sources and resolutions of landslide susceptibility conditioning factors

| Factor | Data source | Resolution / Scale | Year |
| --- | --- | --- | --- |
| Elevation | NASA Earthdata | 12.5 m | 2011 |
| Slope | NASA DEM / ArcGIS Pro | 12.5 m | 2011 |
| Aspect | NASA DEM / ArcGIS Pro | 12.5 m | 2011 |
| Curvature | NASA DEM / ArcGIS Pro | 12.5 m | 2011 |
| Relief | NASA DEM / ArcGIS Pro | 12.5 m | 2011 |
| TWI | NASA DEM / ArcGIS Pro | 12.5 m | 2011 |
| Lithology | National Geological Archives of China | 1:1,000,000 | 2013 |
| Landuse | Esri ArcGIS Living Atlas | 10 m | 2023 |
| NDVI | Sentinel-2 / Google Earth Engine | 10 m | 2019, 2020, 2021, 2023 |
| DTF | OpenStreetMap | 12.5 m (derived) | 2024 |
| DTR | OpenStreetMap | 12.5 m (derived) | 2024 |
| DTW | OpenStreetMap | 12.5 m (derived) | 2024 |

Terrain factors were derived from the 12.5 m DEM using ArcGIS Pro 3.4.3. Distance factors were calculated using Euclidean distance. All conditioning factors were aligned to a 12.5 m modeling grid.

DTF, DTR, and DTW denote distances to faults, roads, and waterways, respectively.

Aspect, Landuse and Lithology retain existing integer codes; other inputs are raw continuous values. IV breaks, smoothing and feature order are copied from the run configurations.

## Negative-Sampling Strategies

|Pool|Recorded sampling rule|
|---|---|
|NS1|Uniform without replacement over the masked candidate domain.|
|NS2|Training-positive-only IV scorer; uniform sampling from the lowest-30% IV pool, including cutoff ties.|
|NS3|50% below baseline Q0.7, 50% in Q0.7–Q0.9.|
|H00|100% below baseline Q0.7.|
|H100|100% in baseline Q0.7–Q0.9.|
|Q50|50% below Q0.5, 50% in Q0.5–Q0.9.|
|Q60|50% below Q0.6, 50% in Q0.6–Q0.9.|
|Q80|50% below Q0.8, 50% in Q0.8–Q0.9.|

Baseline scores are used as candidate-ranking scores, not calibrated future-event probabilities. Known historical buffers, water, boundary and raster validity follow the original code. All 800 historical positives remain; no experiment excluding the 125 July catalogue entries or intersecting historical positives is claimed.

## Train/Test Splits and Random Seeds

There are 18 actual final model runs: NS1/NS2/NS3/H100/Q60 each at 42/43/44; H00/Q50/Q80 only at 42. Missing seeds are marked `not_run` in `manifests/coverage.csv`.

Every run contains 800 positive + 800 negative IDs; canonical combined stratified 70/30 splitting gives 560+560 train and 240+240 test. Saved ID order is preserved. `run_model.py` calls the original split function and checks exact ID membership **and order** against the original run records. It does not regenerate or invent historical IDs. IDs are unique within a run; negative IDs may denote different locations in different runs, so use `(run_key,sample_id)` as the compound key.

The final NS2 wrapper fits IV only on the 560 saved positive training IDs. Baseline RF fit positives for hard pools match their final model's training positives and exclude its 240 test positives; see `manifests/scorer_isolation.csv`. The full historical buffer still uses the inventory as an exclusion design. ID separation does not establish spatial independence.

`pool_ids.csv` means the saved **sampled modeling pool**, not all eligible raster cells. Eligible candidate-cell IDs/masks cannot be reconstructed from these coordinate-free manifests alone. Seeds affect the complete workflow, not only RF initialization.

## How to Run

Run these commands from this repository root. The following first command needs no private data and performs no training:

```sh
python scripts/verify_release.py --self-test
```

The next commands require the data just described:

```sh
python scripts/run_sampling.py --run-key NS1_seed42
python scripts/run_model.py --run-key NS1_seed42
python scripts/run_sampling.py --run-key NS2_seed42
python scripts/run_model.py --run-key NS2_seed42
python scripts/run_sampling.py --run-key H100_seed43
python scripts/run_model.py --run-key H100_seed43
python scripts/evaluate_rasters.py --config configs/evaluation.json
python scripts/event_bootstrap.py --data-root outputs/evaluation --output-root outputs/event_bootstrap
python scripts/spatial_bootstrap.py --events data/evaluation/event_landslide_scores.csv --domain data/evaluation/common_valid_domain.tif --event-bootstrap-root outputs/event_bootstrap --output-root outputs/spatial_bootstrap
```

Sampling and modeling produce timestamped independent run directories and do not automatically rewrite input configurations. After inspecting a PASS sampling run, copy its point files to the explicit `data/samples/...` paths in the model configuration, or pass an edited configuration through `--config`. Supply the matching NS1 RF raster at `data/predictions/NS1_seed<seed>.tif` before hard-pool sampling. Evaluation configurations similarly point to the exact final model rasters; stage them explicitly. Preserve original source point order, attributes and IDs. Do not call the raw NS2 entry point directly for the corrected experiment: it is preserved as source evidence, while `scripts/run_sampling.py` applies the actual training-only IV guard.

`--help` is supported by every script. Evaluation requires a nonempty valid domain with at least 20 pixels so the smallest 5% budget selects cells; the paper domain has 16,500,856. No private-data commands were executed as part of release preparation.

## Evaluation Protocol

The common domain contains 16,500,856 cells at 12.5 m; 22,503 event representatives are used. At budget b, select exactly floor(bN) cells by descending RF score and ascending fixed pixel index. Ties never use event labels. Capture uses the same event denominator; Lift uses actual k/N. SR-AUC is the full pixel-step integral `mean((N - event_rank_zero_based - 1)/N)`, not a four-budget or 1%-step approximation. Delta is the domain share at or above the event median score and is a **post-event** diagnostic.

Event bootstrap uses 10,000 paired event replicates per seed and marginal 95% percentile intervals, master seed 20261023 plus training seed. Spatial bootstrap uses 10,000 paired **occupied-block** replicates at predeclared 1/2/5 km scales, EPSG:32649 and a common origin floored from the domain bounds to 5 km. Each draw repeats all events in selected blocks; the denominator is that replicate's total event count. RNG seed is 202610230 + scale_m*10 + training_seed. No multiplicity adjustment is made. These intervals concern fixed prediction fields, not training-seed variability.

## Reproducing the Paper Results

Without private inputs, `verify_release.py` checks source/result hashes, parses configurations, verifies saved IDs and tests ranking/Lift/SR-AUC on synthetic data. This is **result integrity and lightweight functional validation**, not end-to-end paper reproduction.

`results/seed42_metrics.csv` matches v4 Table 2; `results/seed_results.csv` supports internal seed and external mean/sd tables. `results/budget_audit.csv` supports the tie table. `results/spatial_bootstrap.csv` and `side_by_side_intervals.csv` support Table 3 and appendix endpoints. `event_bootstrap.csv` has 24 comparisons; `spatial_bootstrap.csv` has 72 scale-specific comparisons. `docs/PAPER_CONSISTENCY.md` records the checked scope. Historical full-positive-IV NS2 and intermediate independent-negative-split results are deliberately excluded.

Full resampling requires private event-rank NPZs regenerated by `evaluate_rasters.py`, or the original saved event-rank files with unchanged event ordering. Spatial resampling additionally needs the original event table and valid-domain raster. Models do not need retraining if the final authorized prediction fields are available. Reproducing training requires the original geometry, factors and source sample order, not just public ID manifests.

## Data Availability

Only aggregate experimental results and coordinate-free sampled/split IDs are included. Data licenses and permission to redistribute derived IDs/results require author confirmation. See DATA_AVAILABILITY.md. This is not a claim that the private data can be obtained freely.

## Limitations

One region and one typhoon do not establish spatial or cross-event independence. Factor acquisition dates remain unverified. Occupied-block bootstrap conditions on event-bearing blocks and does not model zero-event areas; scale and origin choices are not optimality claims. H100 deficits remain negative under the reported scales; Q60 improvement depends on seed, budget and scale. Missing hard-family seeds are not synthesized. An unchosen code license and unverified data permissions currently prevent claiming an approved open-source release.

## Citation

Use the final publication's confirmed metadata when available. A provisional citation is:

> Wang, Lu, and Qin, Yuzhang. 2026. Internal Discrimination Does Not Predict Ranking Utility: Negative-Sample Construction for Budget-Constrained Spatial Prioritization. ICAIDM 2026 revision manuscript, v4.

No DOI, acceptance or public repository URL is invented.
