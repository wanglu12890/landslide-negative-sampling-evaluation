"""Windowed all-factor prediction and independent output read-back validation."""
from contextlib import ExitStack
import logging
import numpy as np
import pandas as pd
import rasterio
from .utils import windows, valid_values
from .rf_model import susceptibility_scores
from .output_utils import write_csv


def predict_area(model, paths: dict, reference_name: str, destination, batch_size: int) -> None:
    """Read at most one batch of pixels per factor; preserve the complete reference grid."""
    names = list(model.feature_names_in_)
    if names != list(paths):
        raise ValueError("Spatial feature order differs from model feature order")
    with ExitStack() as stack:
        rasters = {name: stack.enter_context(rasterio.open(paths[name])) for name in names}
        reference = rasters[reference_name]
        profile = dict(driver="GTiff", width=reference.width, height=reference.height,
            count=1, crs=reference.crs, transform=reference.transform, dtype="float32", nodata=-9999,
            compress="deflate", tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
        output = stack.enter_context(rasterio.open(destination, "w", **profile))
        output.update_tags(description="RF-predicted landslide susceptibility score", feature_order=",".join(names),
                           validity="All configured factors valid; no additional spatial exclusions")
        processed, predicted = 0, 0
        for batch, window in enumerate(windows(reference.width, reference.height, batch_size), 1):
            arrays = [rasters[name].read(1, window=window, masked=True) for name in names]
            valid = np.logical_and.reduce([valid_values(array) for array in arrays])
            scores = np.full(valid.shape, -9999, dtype="float32")
            if valid.any():
                features = pd.DataFrame({name: array.data[valid] for name, array in zip(names, arrays)})
                scores[valid] = susceptibility_scores(model, features).astype("float32")
                predicted += int(valid.sum())
            output.write(scores, 1, window=window)
            processed += int(window.width * window.height)
            if batch == 1 or batch % 5 == 0:
                logging.info("Prediction: %.1f%%, %d valid cells, batch %d", 100*processed/(reference.width*reference.height), predicted, batch)
        if predicted == 0:
            raise ValueError("No jointly valid cells across all factors; inspect factor NoData footprints")
        logging.info("Prediction: 100%%; %d valid cells", predicted)


def validate_and_summarize(destination, paths: dict, reference_name: str, batch_size: int, stats_path) -> tuple[dict, dict]:
    """Reopen output and all inputs; compare every cell's joint validity and score range."""
    values = []
    valid_count, nodata_count = 0, 0
    with ExitStack() as stack:
        reference = stack.enter_context(rasterio.open(paths[reference_name]))
        rasters = [stack.enter_context(rasterio.open(path)) for path in paths.values()]
        output = stack.enter_context(rasterio.open(destination))
        checks = dict(crs_match=output.crs == reference.crs, transform_match=output.transform == reference.transform,
            shape_match=output.shape == reference.shape, bounds_match=output.bounds == reference.bounds,
            resolution_match=output.res == reference.res, dtype_match=output.dtypes == ("float32",),
            nodata_match=output.nodata == -9999, single_band=output.count == 1,
            nodata_mask_match=True, finite_score_range_0_1=True)
        if not all(checks.values()):
            raise ValueError(f"Output raster metadata mismatch: {destination}: {checks}")
        for window in windows(output.width, output.height, batch_size):
            valid = np.logical_and.reduce([valid_values(r.read(1, window=window, masked=True)) for r in rasters])
            array = output.read(1, window=window)
            if not np.array_equal(array != -9999, valid):
                raise ValueError(f"Output NoData mask differs from all-factor validity at window {window}: {destination}")
            chunk = array[valid]
            if not np.isfinite(chunk).all() or (chunk < 0).any() or (chunk > 1).any():
                raise ValueError(f"Output scores outside finite [0,1] at window {window}: {destination}")
            if chunk.size:
                values.append(chunk)
            valid_count += chunk.size
            nodata_count += array.size-chunk.size
        if not valid_count:
            raise ValueError("Output contains no valid susceptibility cells")
    # Only one float32 score per valid cell is retained, not the full feature matrix.
    # For this study ~17 million valid cells require ~70 MB for exact quantiles.
    scores = np.concatenate(values)
    values.clear()
    mean, std = float(np.mean(scores, dtype=np.float64)), float(np.std(scores, dtype=np.float64))
    minimum, maximum = float(scores.min()), float(scores.max())
    quantiles = np.quantile(scores, [0.05, 0.25, 0.5, 0.75, 0.95], overwrite_input=True)
    stats = dict(valid_cell_count=int(valid_count), nodata_cell_count=int(nodata_count), min=minimum,
        max=maximum, mean=mean, std=std, median=float(quantiles[2]),
        **{key: float(value) for key, value in zip(["q05", "q25", "q50", "q75", "q95"], quantiles)})
    write_csv(pd.DataFrame(stats.items(), columns=["metric", "value"]), stats_path)
    return checks, stats
