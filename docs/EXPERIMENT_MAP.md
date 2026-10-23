# Paper experiment to implementation map

All eight strategies use `src/skills/lsm_modeling/src/dataset_builder.py` for the final canonical split, `rf_model.py` for RF fitting and `evaluation.py` for internal ROC-AUC. The four original modules live under `src/skills/`.

|Stage/strategy|Actual implementation|Configuration/records|Output in this package|
|---|---|---|---|
|Positive construction|ns1_random_sampling/src/positive_samples.py; centroid with inside fallback|configs/NS1_seed*/sampling.yaml; saved ID manifests|manifests/splits/*/pool_ids.csv (IDs only)|
|NS1 candidate mask and sampling|ns1_random_sampling/src/candidate_cells.py, random_sampling.py|configs/NS1_seed*/sampling.yaml|results/seed_results.csv|
|NS2 candidate/IV|ns2_low_susceptibility_sampling/src/candidate_mask.py, reclassification.py, iv_calculation.py, sampling.py; scripts/run_sampling.py training-only guard|configs/NS2_seed*/sampling.yaml; manifests/scorer_isolation.csv|results/seed_results.csv|
|NS3, H00, H100, Q50, Q60, Q80|ns3_hard_negative_sampling/src/candidate_pool.py, hardness.py, sampling.py; baseline NS1 RF|configs/<pool>_seed*/sampling.yaml; manifests/runs.json|results/seed42_metrics.csv and seed_results.csv|
|RF fitting, score rasters|lsm_modeling/src/rf_model.py, spatial_prediction.py|configs/<pool>_seed*/model.yaml|rasters/models omitted; run IDs and metrics retained|
|Strict area, ties, SR-AUC, Lift, delta|scripts/evaluate_rasters.py; original core from revision prepare_manuscript_assets.py|configs/evaluation.json; supplied authorized rasters/mask/events|results/seed42_metrics.csv, budget_audit.csv|
|Event paired bootstrap|scripts/event_bootstrap.py; original paired_event_bootstrap.py|manifests/event_bootstrap_environment.json|results/event_bootstrap.csv|
|Spatial block bootstrap|scripts/spatial_bootstrap.py; original spatial_block_bootstrap.py|manifests/spatial_bootstrap_environment.json|results/spatial_bootstrap.csv; side_by_side_intervals.csv|
|Random seed and fit-ID isolation|configuration sampling/split/model seeds; guarded entry points|manifests/coverage.csv, scorer_isolation.csv, splits/|18 saved runs; no invented seeds|

`manifests/runs.json` resolves each final paper run to actual original relative code/config/input/run provenance. `input_path_mapping.csv` identifies the private input file references and their new explicit staging paths. Historical run identifiers are not instructions to fetch unavailable remote data.
