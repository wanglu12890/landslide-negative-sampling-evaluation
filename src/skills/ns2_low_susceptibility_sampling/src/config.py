"""Configuration-driven factors, breaks and portable paths."""
from pathlib import Path
import re
import numpy as np
import yaml


def load_config(path: Path):
    """Validate the entire factor schema before accessing input data."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    root = Path(cfg["project_root"]).expanduser()
    root = (path.parent / root).resolve() if not root.is_absolute() else root.resolve()
    cfg["project_root"] = str(root)
    f = cfg["factors"]
    order = f.get("feature_order")
    if not isinstance(order, list) or not order or any(not isinstance(n, str) or not re.fullmatch(r"[\w-]+", n) for n in order):
        raise ValueError("feature_order must be a nonempty list of safe feature names")
    for key in ("continuous", "categorical"):
        if not isinstance(f[key], list) or any(not isinstance(n, str) for n in f[key]):
            raise ValueError(f"factors.{key} must be a list of feature names")
    typed = f["continuous"] + f["categorical"]
    if len(set(order)) != len(order) or len(set(typed)) != len(typed) or set(order) != set(typed) or set(order) != set(f["files"]):
        raise ValueError("feature_order, files and disjoint variable types must describe the same unique factors")
    if cfg["reference_raster"] not in order:
        raise ValueError("reference_raster must name an enabled factor")
    if set(cfg.get("reclassification", {})) != set(f["continuous"]):
        raise ValueError("reclassification must contain exactly the continuous factors")
    for name in f["continuous"]:
        rule = cfg["reclassification"][name]
        breaks = rule.get("breaks")
        if (rule.get("method") != "fixed_breaks" or not isinstance(breaks, list) or not breaks
            or any(type(v) not in (int, float) for v in breaks)
            or not np.isfinite(breaks).all() or not np.all(np.diff(np.asarray(breaks, dtype=float)) > 0)):
            raise ValueError(f"{name}: fixed_breaks requires finite numeric strictly increasing breaks")
    if set(cfg["input"]) != {"boundary", "water", "historical_landslide_buffer"}:
        raise ValueError("input must contain only boundary, water, historical_landslide_buffer")
    for key, value in cfg["data_quality"].items():
        if type(value) is not bool:
            raise ValueError(f"data_quality.{key} must be boolean")
    if not cfg["data_quality"]["stop_on_raster_misalignment"]:
        raise ValueError("Grid mismatch cannot be bypassed; stop_on_raster_misalignment must be true")
    smooth = cfg["information_value"]["smoothing"]
    if type(smooth["enabled"]) is not bool or type(smooth["alpha"]) not in (int, float) or not np.isfinite(smooth["alpha"]) or smooth["alpha"] <= 0:
        raise ValueError("smoothing requires boolean enabled and finite alpha > 0")
    low = cfg["low_susceptibility"]
    if low["method"] != "quantile" or type(low["quantile"]) not in (int, float) or not 0 < low["quantile"] <= 1:
        raise ValueError("low_susceptibility requires quantile with 0 < quantile <= 1")
    s = cfg["sampling"]
    if s["sample_count_mode"] not in ("auto", "manual") or s["one_negative_per_raster_cell"] is not True:
        raise ValueError("Sampling requires auto/manual and one_negative_per_raster_cell: true")
    if s["sample_count_mode"] == "manual" and (type(s["manual_sample_count"]) is not int or s["manual_sample_count"] <= 0):
        raise ValueError("manual_sample_count must be a positive integer")
    if type(s["random_seed"]) is not int or not 0 <= s["random_seed"] < 2**63:
        raise ValueError("random_seed must be an integer in [0, 2**63)")
    p = cfg.setdefault("processing", {})
    p.setdefault("chunk_rows", 128)
    p.setdefault("write_iv_factors", True)
    if type(p["chunk_rows"]) is not int or p["chunk_rows"] <= 0 or type(p["write_iv_factors"]) is not bool:
        raise ValueError("Invalid processing chunk_rows/write_iv_factors")
    diag = cfg.setdefault("diagnostics", {})
    for key, default in (("low_landslide_count_threshold", 3), ("small_area_ratio_threshold", 0.001), ("extreme_iv_abs_threshold", 5.0)):
        diag.setdefault(key, default)
        if type(diag[key]) not in (int, float) or not np.isfinite(diag[key]) or diag[key] < 0:
            raise ValueError(f"diagnostics.{key} must be finite and nonnegative")
    factors = {n: (root / f["directory"] / f["files"][n]).resolve() for n in order}
    vectors = {n: (root / v).resolve() for n, v in cfg["input"].items()}
    positive = (root / cfg["positive_samples"]["path"]).resolve()
    compare = cfg.get("comparison", {}).get("ns1_negative_samples")
    compare = (root / compare).resolve() if compare else None
    output = (root / cfg["output"]["root"]).resolve()
    if any(output == v.parent or output.is_relative_to(v.parent) for v in [*factors.values(), *vectors.values(), positive]):
        raise ValueError("Output must be outside all source-data directories")
    return cfg, factors, vectors, positive, compare, output
