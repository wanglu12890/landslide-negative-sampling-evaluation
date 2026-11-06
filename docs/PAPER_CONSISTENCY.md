# Manuscript and repository consistency

## Numerical comparison record

The earlier v4 consistency checks found that all eight seed-42 metric rows and all 24 paired comparison rows, including event-level and 1/2/5 km spatial-block interval endpoints, matched the checked v4 LaTeX at its printed precision. Additional checks covered five internal-seed summary rows, five external mean/standard-deviation rows, and eight tie-audit rows. No numerical discrepancies were found within that checked scope.

These statements document the earlier numerical comparison; they do not imply a new comparison against every subsequent manuscript edit. Numeric source tables were copied from the recorded experiments, not fitted or recomputed during release preparation.

Result files are identified by content rather than manuscript table number. See the README for the mapping between aggregate result files and manuscript analyses. Complete seed-specific intervals are available in the repository CSV files.

## Manifest completeness and functional validation

The updated public download contains all 72 required manifest CSV files for the 18 recorded runs. Each run includes `pool_ids.csv`, `train_ids.csv`, `test_ids.csv`, and `scorer_fit_positive_ids.csv`.

A fresh download passed:

    python scripts/verify_release.py --self-test

The checks cover saved IDs and configurations, canonical split membership and order, scorer-fit positive-ID separation, source/result hashes, aggregate-result consistency, command-line interfaces, and synthetic ranking and spatial-block calculations. This is integrity and lightweight functional validation, not end-to-end reproduction with private geospatial inputs.

## Interpretation and reproduction boundaries

- `pool_ids.csv` represents the saved sampled modeling pool, not every eligible raster cell in the negative-candidate domain. Candidate-cell masks remain private inputs or outputs.
- Historical NS2 models using full-positive IV fitting and intermediate models using independent negative splitting are excluded; only the final recorded runs are mapped.
- Supervised pool scorers use the 560 training positives. The 25 m exclusion mask uses all 800 historical polygons, including those later assigned to internal testing. Scorer-ID separation therefore does not establish strict training-only spatial isolation.
- Event evaluation uses a temporally held-out inventory and does not establish strict spatial or cross-event independence.
- All 800 historical positives are retained. Of these, 125 landslides were identified for the period from December 2023 to 24 July 2024 using PlanetScope imagery acquired on 24 July 2024 as the base image.
- No exclusion sensitivity analysis was conducted for these 125 pre-event historical landslides or for historical polygons overlapping the post-event inventory. Robustness to historical/event inventory overlap remains untested.
- Event-level and spatial-block intervals describe fixed-prediction event-sampling uncertainty; training-seed variability is reported separately.
- DTF, DTR, and DTW denote distances to faults, roads, and waterways, respectively. Conditioning-factor sources and resolutions are documented in the manuscript and README.
- The repository code is released under the MIT License. The code license does not establish redistribution rights for private geospatial inputs or derived records.
- Full reproduction requires the authorized private inputs described in the README and `DATA_AVAILABILITY.md`, including original geometries and sample order, aligned factor rasters, prediction fields, the evaluation mask, and event representative-point data.
