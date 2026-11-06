# Porting notes

## Source modules and configurations

1. Original NS1/NS2/NS3/RF entry modules and their required `src` modules were copied byte-for-byte into `src/skills/`. The package layout preserves `skills.*` cross-module imports. No unrelated business modules, original test suites, cached bytecode or third-party implementations are vendored. `manifests/file_provenance.csv` records original workspace-relative paths and hashes.

2. Actual configuration snapshots were adapted through structured YAML parsing. `project_root` becomes `../..` from each per-run configuration folder; input paths become explicit `data/` parameters and output paths become `outputs/<run_key>/...`. Quantiles, seeds, class ratios, fixed IV breaks, RF parameters and split flags are retained. Source/target path mappings are recorded in `manifests/input_path_mapping.csv`. Configurations require explicit input staging and do not search for the latest run.

## Sampling and model fitting

3. `scripts/run_sampling.py` applies the corrected NS2 training-ID filtering used in the final experiment wrapper: retain all 800 output positives, but pass only the 560 recorded training positives into the original IV implementation. Hard-pool sampling calls the original module with the matching seed's baseline score configuration. The original raw NS2 code is retained as source evidence and is not the corrected experiment entry point.

4. `scripts/run_model.py` invokes the original canonical combined stratified split and checks exact saved ID membership and order. The intermediate independent-negative-split wrapper is not used. Original RF training and prediction remain unchanged; no tuning or algorithmic repair was introduced during extraction.

5. Training-only supervised scorer fitting is distinct from candidate-mask construction. The 25 m exclusion mask uses all 800 historical polygons, including those later assigned to internal testing. Scorer-fit ID separation does not establish strict training-only spatial isolation.

## Evaluation and resampling

6. Event-level and spatial-block bootstrap scripts retain the original RNG, iteration counts, pairing, estimators, quantiles and scale choices. Hardcoded workstation paths and report destinations were adapted to command-line parameters. RNG and resampling code blocks were compared directly with their source implementations during porting.

7. The raster evaluator extracts the existing lexsort, rank, capture, Lift, full-step success-rate AUC and delta calculations. Plotting and the hardcoded eight-raster mapping were removed, and explicit grid/domain checks were added. These adaptations do not change the recorded evaluation definitions.

## Manifests and aggregate results

8. Saved sample IDs and row order were column-extracted, not regenerated. CSV serialization changes file bytes but not the included values. Source train/test hashes identify the original full private tables. `(run_key, sample_id)` is the public compound identity. `pool_ids.csv` represents the sampled modeling dataset, not all eligible raster cells.

9. The completed release contains 72 manifest CSV files for 18 recorded runs. Each run includes `pool_ids.csv`, `train_ids.csv`, `test_ids.csv`, and `scorer_fit_positive_ids.csv`. Missing manifests were restored from the original release package without regenerating IDs or rerunning experiments. Unrun seed configurations remain marked `not_run`.

10. Result CSVs were copied with only absolute run/raster path columns removed. Raw logs and large objects are excluded. Original run metadata was reduced to relevant metrics, versions and QC status; complete private execution logs are not redistributed. `manifests/runs.json` is an extraction of recorded experiments, not a newly executed training record.

## Validation and licensing

11. Release preparation did not edit the original project files or silently correct scientific algorithms. Verification uses synthetic inputs under ignored `outputs/`; temporary synthetic inputs are removed after testing.

12. A fresh download of the completed release passed `python scripts/verify_release.py --self-test`. This checks included manifests, configurations, split order, scorer-fit positive-ID separation, source/result hashes, aggregate-result consistency, command-line interfaces and lightweight synthetic calculations. It does not constitute end-to-end reproduction with private geospatial inputs.

13. Dependency version records are preserved. A clean environment installation was not performed during extraction, and the later release self-check does not establish that every recorded training or statistics environment has been independently recreated.

14. The repository code is released under the MIT License; see `LICENSE`. The code license does not establish redistribution rights for private geospatial inputs or derived records. Required private inputs and reproduction boundaries are documented in the README and `DATA_AVAILABILITY.md`.
