"""Load original positives and spatial constraints without moving sample points."""
import logging
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from rasterio.transform import rowcol
from skills.ns1_random_sampling.src.io_utils import clean_layer
from skills.lsm_modeling.src.feature_extraction import extract_features
from .output_utils import write_csv
from .utils import centers


def load_vectors(paths: dict, reference: dict, report: dict) -> dict:
    unions = {}
    report["geometry"] = {}
    for key, path in paths.items():
        layer = gpd.read_file(path)
        if layer.crs is None or layer.crs != reference["crs"]:
            raise ValueError(f"{key}: {path}: CRS {layer.crs} != reference {reference['crs']}")
        cleaned, stats = clean_layer(layer, key)
        report["geometry"][key] = {**stats, "crs": str(layer.crs)}
        if stats["excluded_source_fids"] or (cleaned.empty and key != "water"):
            raise ValueError(f"{key}: {path}: empty/unusable polygon mask; provide complete existing geometry")
        unions[key] = shapely.union_all(cleaned.geometry.to_numpy())
        shapely.prepare(unions[key])
        logging.info("Vector %s: %s", key, stats)
    return unions


def load_positive(path, reference, factors, unions, cfg, run, report):
    """Keep every original positive; optionally omit invalid positives from IV counts only."""
    original = gpd.read_file(path)
    if original.empty or original.crs is None or original.crs != reference["crs"]:
        raise ValueError(f"{path}: empty positive layer or CRS mismatch")
    if "label" not in original or not original.label.eq(1).all():
        raise ValueError(f"{path}: positive labels must all equal 1")
    if original.geometry.isna().any() or original.geometry.is_empty.any() or not original.geom_type.eq("Point").all():
        raise ValueError(f"{path}: positive geometry must be nonempty Point")
    x, y = original.geometry.x.to_numpy(), original.geometry.y.to_numpy()
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError(f"{path}: non-finite coordinates")
    rows, cols = (np.asarray(a, dtype="int64") for a in rowcol(reference["transform"], x, y))
    inside = (rows >= 0) & (cols >= 0) & (rows < reference["height"]) & (cols < reference["width"])
    field = "positive_id" if "positive_id" in original else "sample_id"
    if field not in original or original[field].isna().any() or original[field].duplicated().any():
        raise ValueError(f"{path}: positive_id/sample_id must be present, non-null and unique")
    points = pd.DataFrame(dict(sample_id=original[field].astype(str), label=1, x=x, y=y, row=rows, col=cols))
    extracted, bad = extract_features(points.loc[inside].reset_index(drop=True), factors,
                                      run / "qc/invalid_positive_samples.csv", report)
    invalid = pd.read_csv(run / "qc/invalid_positive_samples.csv")
    extra = []
    for p in points.loc[~inside].itertuples():
        extra.append(dict(sample_id=p.sample_id, label=1, x=p.x, y=p.y, invalid_factor="domain", invalid_value=None, reason="outside_raster"))
    cx, cy = centers(reference["transform"], extracted.row.to_numpy(), extracted.col.to_numpy())
    domain_ok = shapely.intersects_xy(unions["boundary"], cx, cy) & ~shapely.intersects_xy(unions["water"], cx, cy)
    for p in extracted.loc[~domain_ok].itertuples():
        extra.append(dict(sample_id=p.sample_id, label=1, x=p.x, y=p.y, invalid_factor="domain", invalid_value=None, reason="pixel_center_outside_IV_background"))
    if extra:
        invalid = pd.concat([invalid, pd.DataFrame(extra)], ignore_index=True)
    write_csv(invalid, run / "qc/invalid_positive_samples.csv")
    valid = extracted.loc[~bad & domain_ok].copy()
    report["positive"] = dict(path=str(path), positive_count=len(original), valid_count=len(valid),
        invalid_count=len(original)-len(valid), duplicate_pixel_count=int(points.duplicated(["row", "col"]).sum()),
        output_policy="All original positives preserved; IV uses valid positives only")
    if len(valid) != len(original) and cfg["data_quality"]["stop_on_invalid_positive_sample"]:
        raise ValueError("Invalid positives; inspect qc/invalid_positive_samples.csv; no positions regenerated")
    if valid.empty:
        raise ValueError("No valid positives for IV calculation")
    return original, points, valid
