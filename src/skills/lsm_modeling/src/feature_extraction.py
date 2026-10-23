"""Point checks and raw same-cell factor extraction."""
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from rasterio.windows import Window
from .output_utils import write_csv
from .utils import class_counts, valid_values


def load_samples(paths: dict, reference: dict, allow_imbalanced: bool, report: dict) -> pd.DataFrame:
    """Read only the two configured point sources; never generate or repair samples."""
    tables, diagnostics, errors = [], {}, []
    for kind, expected in (("positive", 1), ("negative", 0)):
        path = paths[f"{kind}_samples"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing {kind} sample file: {path}; check YAML")
        frame = gpd.read_file(path)
        if frame.crs is None or frame.crs != reference["crs"]:
            raise ValueError(f"Sample CRS mismatch/missing: {path}: {frame.crs}; reference: {reference['crs']}")
        if frame.empty or frame.geometry.isna().any() or frame.geometry.is_empty.any() or not (frame.geom_type == "Point").all():
            raise ValueError(f"{path}: samples must be nonempty Point geometries; check source samples")
        if "label" in frame and not frame.label.eq(expected).all():
            raise ValueError(f"{path}: label must equal {expected} for every {kind} sample")
        x, y = frame.geometry.x.to_numpy(), frame.geometry.y.to_numpy()
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError(f"{path}: non-finite point coordinates")
        rows, cols = rowcol(reference["transform"], x, y)
        rows, cols = np.asarray(rows), np.asarray(cols)
        outside = (rows < 0) | (cols < 0) | (rows >= reference["height"]) | (cols >= reference["width"])
        duplicates = int(pd.DataFrame(dict(x=x, y=y)).duplicated().sum())
        diagnostics[kind] = dict(path=str(path), count=len(frame), duplicate_geometry_count=duplicates,
                                  outside_raster_count=int(outside.sum()), crs=str(frame.crs))
        if outside.any() or duplicates:
            errors.append(f"{path}: outside raster={int(outside.sum())}, duplicate coordinates={duplicates}; inspect source samples")
        id_field = next((field for field in ("sample_id", f"{kind}_id") if field in frame), None)
        ids = frame[id_field].astype(str) if id_field else pd.Series([f"{kind}_{i+1:06d}" for i in range(len(frame))])
        if id_field and (frame[id_field].isna().any() or ids.eq("").any()):
            errors.append(f"{path}: missing sample IDs")
        tables.append(pd.DataFrame(dict(sample_id=ids.to_numpy(), label=expected, x=x, y=y,
                                       row=rows, col=cols, sample_type=kind)))
    result = pd.concat(tables, ignore_index=True)
    overlap = int(result.duplicated(["x", "y"]).sum())
    report["samples"] = dict(source_checks=diagnostics, **class_counts(result),
        positive_negative_ratio=f"{len(tables[0])}:{len(tables[1])}",
        cross_source_coordinate_overlap=overlap, allow_imbalanced_samples=allow_imbalanced)
    if overlap:
        errors.append("Positive/negative sample sources have identical coordinates; inspect source files")
    if result.sample_id.duplicated().any():
        errors.append("Sample IDs must be globally unique across both files")
    if len(tables[0]) != len(tables[1]) and not allow_imbalanced:
        errors.append("P:N is not 1:1. No resampling performed; set allow_imbalanced_samples: true only if intended")
    if errors:
        raise ValueError("; ".join(errors))
    return result


def extract_features(samples: pd.DataFrame, paths: dict, invalid_path, report: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """Read exact integer row/col values, preserving original raster numbers."""
    result = samples.copy()
    invalid, bad_samples = [], np.zeros(len(samples), dtype=bool)
    for factor, path in paths.items():
        values = []
        with rasterio.open(path) as raster:
            for index, sample in samples.iterrows():
                array = raster.read(1, window=Window(int(sample.col), int(sample.row), 1, 1), masked=True)
                value = array.data[0, 0]
                good = bool(valid_values(array)[0, 0])
                values.append(value if good else np.nan)
                if not good:
                    bad_samples[index] = True
                    reason = "NoData_or_masked" if np.ma.getmaskarray(array)[0, 0] else "NaN_or_Inf"
                    invalid.append(dict(sample_id=sample.sample_id, label=sample.label, x=sample.x, y=sample.y,
                                        invalid_factor=factor, invalid_value=value, reason=reason))
        result[factor] = values
    write_csv(pd.DataFrame(invalid, columns=["sample_id", "label", "x", "y", "invalid_factor", "invalid_value", "reason"]), invalid_path)
    report["extraction"] = dict(valid_sample_count=int((~bad_samples).sum()), invalid_sample_count=int(bad_samples.sum()),
                                invalid_factor_occurrence_count=len(invalid), method="raw raster row/col values; no interpolation")
    return result, bad_samples
