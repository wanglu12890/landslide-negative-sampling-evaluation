# Authorized private input schema

- Model geometry: original Point GeoPackage with CRS matching factors and `label` equal to 1/0, stable `sample_id` or `positive_id`/`negative_id`, and original row order. Point coordinates are required locally but omitted from the public manifests.
- NS1: original historical polygons, water, boundary and 25 m historical-buffer polygons; original temporal/state attributes are required for the NS3-family inventory checks. See the actual per-run YAML for exact temporal fields/states.
- Factors: aligned single-band 12.5 m rasters with exact original nodata/masks and codes. The original loaders stop on misalignment or invalid values; do not resample silently to bypass checks.
- Raster evaluation: common valid-domain GeoTIFF with value 1 for included pixels; event CSV with `row,col,common_valid` and original event row order. Actual `event_id,x,y` are required additionally for spatial bootstrap. Evaluation NPZ schema is `event_rank,event_row,event_col,event_index`; ranks are zero-based.
- Event bootstrap: all NS3/H100/Q60 NPZ files for seeds 42/43/44, named `<pool>_seed<seed>_event_ranks.npz`, in `--data-root`. The source implementation asserts 22503 events and uses the recorded common domain count. It is a fixed-paper experiment script, not a generic n-event library.
- Spatial bootstrap: the original domain raster/events and the previous event-bootstrap output directory, which contains `seed*_event_paired_differences.csv`. Each repeated occupied block contributes all its events; the denominator varies with draws.

Never publish these supplied private input files merely because the repository's code can read them. Sampling outputs must be staged explicitly into the static configuration paths before model training. Newly regenerated sampling IDs and model split IDs are checked against saved manifests; a mismatch stops the run instead of silently inventing correspondence.
