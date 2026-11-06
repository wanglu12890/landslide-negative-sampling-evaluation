# Public release audit

## Current release status

The repository is publicly available at:

https://github.com/wanglu12890/landslide-negative-sampling-evaluation

The repository code is released under the MIT License; see `LICENSE`. This code license does not establish redistribution rights for private geospatial inputs or derived records.

The updated public download contains all 72 required manifest CSV files for the 18 recorded runs. A fresh download passed:

    python scripts/verify_release.py --self-test

This provides manifest, configuration, source/result integrity and lightweight functional validation. It does not constitute end-to-end reproduction with private geospatial inputs or a determination of data redistribution rights.

## Included and excluded materials

| Class | Handling | Boundary or qualification |
| --- | --- | --- |
| Original implementation | Necessary NS1/NS2/NS3/RF packages, with source paths and hashes retained | The repository code is distributed under MIT; upstream provenance records are preserved. |
| Sample IDs | Saved IDs, class labels and record order only; identifiers are run-qualified | IDs contain no coordinates or feature values. Negative IDs are not globally spatial identifiers; use `(run_key, sample_id)`. |
| Manifest sets | Four files for each of 18 recorded runs, totaling 72 CSV files | Missing files were restored from the original release package without regenerating IDs or rerunning experiments. |
| Results | Aggregate numeric tables and recorded code/environment information | Publication of aggregate results does not grant access to private source data. |
| Coordinates and features | Original sample/event coordinates and factor values are excluded | CRS, block scales and the coarse grid origin remain as methodological reproducibility parameters. |
| Imagery and geographic datasets | Excluded | Access and redistribution remain subject to the applicable data owners' terms. |
| Models, predictions and large caches | Excluded | Full evaluation reproduction requires the corresponding authorized prediction fields and other private inputs. |
| Logs and private metadata | Raw logs and source XML are excluded; reduced run status, metrics and version records are retained | Reduced summaries are extracted records, not complete execution logs. |
| Credentials and local connection information | Text-pattern, extension and size checks were performed during the original release preparation | These historical checks are limited to the files and patterns inspected; they are not a guarantee that every later revision has been scanned. |
| Local absolute paths | Configuration paths were adapted; absolute run/raster path columns were removed from copied results | Relative original run identifiers remain for provenance. |
| Third-party dependencies | Referenced through requirements files; dependency implementations are not vendored | Dependencies retain their own licenses. |
| Code license | MIT License in the repository root | The code license does not establish redistribution rights for private geospatial inputs or derived records. |

## Validation record and scope

The completed public download passed checks covering:

- Python syntax and source module imports.
- Saved IDs, configurations and canonical train/test split membership and order.
- Supervised scorer-fit positive-ID separation.
- Source and result file hashes.
- Aggregate-result consistency.
- Command-line help.
- Synthetic exact-area ranking, tied scores, Lift and complete success-rate AUC.
- Paired spatial-block calculations with repeated blocks and variable denominators.

These checks do not retrain the models, repeat the reported event/spatial bootstrap experiments, recreate a clean dependency environment, or reproduce the experiments from private geospatial inputs.

`PUBLIC_RELEASE_SCAN.json` and `VALIDATION.json` document their respective recorded checks. Historical scan or validation records must not be interpreted as newly executed checks of every subsequent repository edit.

## Manuscript interpretation and data boundary

Event evaluation uses a temporally held-out inventory and does not establish strict spatial or cross-event independence.

Supervised pool scorers use the 560 training positives. The 25 m exclusion mask uses all 800 historical polygons, including those later assigned to internal testing. Scorer-ID separation therefore does not establish strict training-only spatial isolation.

All 800 historical positives are retained. Of these, 125 landslides were identified for the period from December 2023 to 24 July 2024 using PlanetScope imagery acquired on 24 July 2024 as the base image. No exclusion sensitivity analysis was conducted for these landslides or for historical polygons overlapping the post-event inventory. Robustness to historical/event inventory overlap remains untested.

Conditioning-factor sources, resolutions and available acquisition years are documented in the manuscript and README. Private source metadata XML is not redistributed; the public source table does not provide the underlying private rasters or their complete metadata.

See `PAPER_CONSISTENCY.md` for the scope of the manuscript comparison, and the README and `DATA_AVAILABILITY.md` for required private inputs and reproduction limits.
