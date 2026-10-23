# Data availability and reproduction boundary

## Included, subject to author release approval

Aggregate metrics and bootstrap intervals; run-local positive/negative sampled IDs and saved train/test ID order; relative run identifiers, file hashes, code and portable configurations. IDs were extracted from saved tables, not newly randomized. They contain no coordinate columns, raster indices, feature values or source geometry. Still confirm permission for these derived records before publication.

## Not redistributed

Remote-sensing imagery, factor rasters, original vector inventories and attributes, polygons/point coordinates, water/boundary/buffer layers, event-level ranks/differences, event-to-block maps, serialized models, prediction rasters, private source metadata XML, workstation logs and Git history. Their upstream licenses and redistribution permissions are unknown here. No download endpoint or commercial-data entitlement is assumed.

## What can run

1. Independently with included files: integrity checks, ID/count/config consistency and synthetic ranking/cluster smoke tests.
2. With authorized prediction fields plus common-domain mask and event representatives: exact-area ranking and event/spatial bootstrap.
3. With original factors, geometry and stable point/feature order: sampling and RF training, guarded by original IDs.
4. Not available from this package alone: an end-to-end exact replication, exhaustive candidate-cell pool IDs, missing seed runs, or acquisition-source metadata.

Stage user-owned inputs under `data/` as documented in the configurations. Generated samples/predictions are private outputs; they are not automatically staged or approved for release. The coarse spatial grid origin retained in statistics metadata is a reproducibility parameter, not an event coordinate list.
