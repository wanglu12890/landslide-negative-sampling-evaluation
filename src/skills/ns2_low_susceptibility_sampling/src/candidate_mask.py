"""Distinct IV background and buffered negative candidate domains."""
from collections import Counter
import logging
import numpy as np
import pandas as pd
import rasterio
import shapely
from .utils import centers, factor_windows, profile, valid_values, windows
from .reclassification import classify
from .output_utils import write_csv


def domains(arrays, window, reference, unions):
    """Pixel-center rules identical to NS1; buffer excluded only from candidates."""
    valid = np.logical_and.reduce([valid_values(a) for a in arrays.values()])
    rows = np.arange(window.row_off, window.row_off+window.height)[:, None]
    cols = np.arange(reference["width"])[None, :]
    x, y = centers(reference["transform"], rows, cols)
    boundary = shapely.intersects_xy(unions["boundary"], x, y)
    water = shapely.intersects_xy(unions["water"], x, y)
    buffer = shapely.intersects_xy(unions["historical_landslide_buffer"], x, y)
    background = valid & boundary & ~water
    return valid, background, background & ~buffer


def build_background(paths, reference, unions, cfg, run, report):
    """Count background classes and write base mask in one streaming pass."""
    counts = {name: Counter() for name in paths}
    observed = {name: set() for name in paths}
    totals = dict(factor_valid_cells=0, iv_background_cells=0, base_candidate_cells=0)
    with rasterio.open(run / "spatial/base_candidate_mask.tif", "w", **profile(reference, "uint8", 255)) as output:
        for i, (window, arrays) in enumerate(factor_windows(paths, cfg["reference_raster"], cfg["processing"]["chunk_rows"])):
            valid, background, base = domains(arrays, window, reference, unions)
            totals["factor_valid_cells"] += int(valid.sum())
            totals["iv_background_cells"] += int(background.sum())
            totals["base_candidate_cells"] += int(base.sum())
            mask = base.astype("uint8")
            mask[~valid] = 255
            output.write(mask, 1, window=window)
            for name, array in arrays.items():
                codes = classify(array.data[background], name, cfg)
                unique, nums = np.unique(codes, return_counts=True)
                counts[name].update({int(k): int(v) for k, v in zip(unique, nums)})
                if name in cfg["factors"]["categorical"]:
                    observed[name].update(int(v) for v in np.unique(classify(array.data[valid_values(array)], name, cfg)))
            if i % 10 == 0:
                logging.info("Background rows %d/%d", window.row_off+window.height, reference["height"])
    report["domains"] = totals
    if not totals["iv_background_cells"] or not totals["base_candidate_cells"]:
        raise ValueError(f"Empty IV background/base candidate; check factors and vector masks: {totals}")
    return counts, observed


def build_low_mask(reference, cfg, run, report):
    """Compute quantiles from saved float32 IV; candidate threshold uses base cells only."""
    all_values, base_values = [], []
    chunk_rows = cfg["processing"]["chunk_rows"]
    with rasterio.open(run / "spatial/IV_total.tif") as iv, rasterio.open(run / "spatial/base_candidate_mask.tif") as base:
        for w in windows(iv.width, iv.height, chunk_rows):
            a, b = iv.read(1, window=w), base.read(1, window=w)
            all_values.append(a[a != -9999])
            base_values.append(a[b == 1])
    values = np.concatenate(all_values)
    candidates = np.concatenate(base_values)
    all_values.clear()
    base_values.clear()
    if not values.size or not candidates.size or not np.isfinite(values).all() or not np.isfinite(candidates).all():
        raise ValueError("No finite IV values for background/candidate quantile")
    qs = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    stats = dict(valid_cell_count=len(values), min=float(values.min()), max=float(values.max()),
                 mean=float(values.mean(dtype="float64")), std=float(values.std(dtype="float64")))
    stats.update({f"q{int(q*100):02d}": float(v) for q, v in zip(qs, np.quantile(values, qs, overwrite_input=True))})
    stats["median"] = stats["q50"]
    q = cfg["low_susceptibility"]["quantile"]
    threshold = float(np.quantile(candidates, q, overwrite_input=True))
    summary = dict(base_candidate_cells=len(candidates), quantile=q, iv_quantile_threshold=threshold,
                   IV_min=float(candidates.min()), IV_max=float(candidates.max()))
    summary[f"IV_Q{q*100:g}"] = threshold
    del values, candidates
    low_count = 0
    with (rasterio.open(run / "spatial/IV_total.tif") as iv,
          rasterio.open(run / "spatial/base_candidate_mask.tif") as base,
          rasterio.open(run / "spatial/low_susceptibility_mask.tif", "w", **profile(reference, "uint8", 255)) as output):
        for w in windows(iv.width, iv.height, chunk_rows):
            a, b = iv.read(1, window=w), base.read(1, window=w)
            low = (b == 1) & (a.astype("float64") <= threshold)
            mask = low.astype("uint8")
            mask[b == 255] = 255
            output.write(mask, 1, window=w)
            low_count += int(low.sum())
    summary.update(low_candidate_cells=low_count, low_candidate_ratio=low_count/summary["base_candidate_cells"])
    report.update(iv_total=stats, low_susceptibility=summary)
    write_csv(pd.DataFrame(stats.items(), columns=["metric", "value"]), run / "qc/iv_total_statistics.csv")
    write_csv(pd.DataFrame([summary]), run / "qc/low_susceptibility_summary.csv")
    return threshold, low_count
