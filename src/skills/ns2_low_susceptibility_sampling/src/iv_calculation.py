"""Information value with explicit additive positive-count smoothing."""
from contextlib import ExitStack
import logging
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from .reclassification import classify, class_description
from .candidate_mask import domains
from .utils import factor_windows, profile
from .output_utils import write_csv


def calculate_iv(counts, observed, positives, cfg, reference, run, report):
    """K counts background-present classes only; empty classes have no IV."""
    alpha = cfg["information_value"]["smoothing"]["alpha"] if cfg["information_value"]["smoothing"]["enabled"] else 0.0
    total_l = len(positives)
    rows, lookup, summaries = [], {}, {}
    crs = CRS(reference["crs"])
    if not crs.is_projected:
        raise ValueError("Area reporting requires projected reference CRS")
    t = reference["transform"]
    pixel_area = abs(t.a*t.e-t.b*t.d) * crs.axis_info[0].unit_conversion_factor * crs.axis_info[1].unit_conversion_factor
    failures = []
    for name in cfg["factors"]["feature_order"]:
        total_a = sum(counts[name].values())
        k = len(counts[name])
        positive_counts = pd.Series(classify(positives[name].to_numpy(), name, cfg)).value_counts().to_dict()
        continuous = name in cfg["factors"]["continuous"]
        codes = range(1, len(cfg["reclassification"][name]["breaks"])+2) if continuous else sorted(observed[name] | set(positive_counts))
        lookup[name] = {}
        start = len(rows)
        for code in codes:
            a, l = counts[name].get(code, 0), int(positive_counts.get(code, 0))
            empty = a == 0
            pa = a/total_a
            pl = (l+alpha)/(total_l+alpha*k) if not empty else np.nan
            iv = float(np.log(pl/pa)) if a and pl > 0 else (np.nan if empty else -np.inf)
            row = dict(factor=name, factor_type="continuous" if continuous else "categorical", class_id=int(code),
                **class_description(name, code, cfg), cell_count=a, area_km2=a*pixel_area/1e6, area_ratio=pa,
                landslide_count=l, landslide_ratio_raw=l/total_l, landslide_ratio_smoothed=pl, iv_value=iv,
                empty_class=empty, smoothing_alpha=alpha, effective_class_count=k,
                zero_landslide_class=l == 0, low_landslide_count=l < cfg["diagnostics"]["low_landslide_count_threshold"],
                small_area_class=pa < cfg["diagnostics"]["small_area_ratio_threshold"],
                extreme_iv=bool(np.isfinite(iv) and abs(iv) > cfg["diagnostics"]["extreme_iv_abs_threshold"]),
                nonfinite_iv=bool(not empty and not np.isfinite(iv)))
            rows.append(row)
            if empty and l:
                failures.append(f"{name} class {code}: positives in absent background class")
            if empty and cfg["data_quality"]["stop_on_empty_class"]:
                failures.append(f"{name} class {code}: empty class and stop_on_empty_class=true")
            if not empty:
                if not np.isfinite(iv):
                    failures.append(f"{name} class {code}: nonfinite IV; inspect smoothing setting/zero count")
                lookup[name][int(code)] = iv
        subset = rows[start:]
        finite = [r["iv_value"] for r in subset if not r["empty_class"] and np.isfinite(r["iv_value"])]
        summaries[name] = dict(actual_class_count=k, configured_breaks=cfg["reclassification"].get(name, {}).get("breaks"),
            landslide_counts={str(r["class_id"]): r["landslide_count"] for r in subset},
            iv_min=min(finite) if finite else None, iv_max=max(finite) if finite else None,
            **{flag: [r["class_id"] for r in subset if r[flag]] for flag in
               ("empty_class", "zero_landslide_class", "low_landslide_count", "small_area_class", "extreme_iv", "nonfinite_iv")})
    table = pd.DataFrame(rows)
    write_csv(table, run / "qc/iv_class_statistics.csv")
    report["iv_factors"] = summaries
    report["smoothing"] = dict(enabled=alpha > 0, alpha=alpha, formula="ln(((L_j+alpha)/(L+alpha*K))/(A_j/A)); K=A_j>0 classes")
    if failures:
        raise ValueError("; ".join(failures))
    return lookup, table


def map_iv(values, factor, cfg, lookup):
    """Apply exact code-to-IV mapping; never invent values for absent classes."""
    codes = classify(values, factor, cfg)
    keys = np.asarray(sorted(lookup[factor]), dtype="int64")
    index = np.searchsorted(keys, codes)
    if (index >= len(keys)).any() or not np.array_equal(keys[index], codes):
        raise ValueError(f"{factor}: observed class has no IV mapping")
    return np.asarray([lookup[factor][int(k)] for k in keys], dtype="float64")[index]


def build_iv_surface(paths, reference, unions, cfg, lookup, run):
    """Sum configured IVs in float64 order, saving float32 screening surfaces."""
    with ExitStack() as stack:
        total_out = stack.enter_context(rasterio.open(run / "spatial/IV_total.tif", "w", **profile(reference, "float32", -9999)))
        total_out.update_tags(role="negative-sample screening surface", feature_order=",".join(paths))
        individual = {n: stack.enter_context(rasterio.open(run / f"spatial/iv_factors/IV_{n}.tif", "w", **profile(reference, "float32", -9999)))
                      for n in paths} if cfg["processing"]["write_iv_factors"] else {}
        for i, (window, arrays) in enumerate(factor_windows(paths, cfg["reference_raster"], cfg["processing"]["chunk_rows"])):
            _, background, _ = domains(arrays, window, reference, unions)
            total = np.zeros(int(background.sum()), dtype="float64")
            for name, array in arrays.items():
                values = map_iv(array.data[background], name, cfg, lookup)
                total += values
                if name in individual:
                    mapped = np.full(background.shape, -9999, dtype="float32")
                    mapped[background] = values
                    individual[name].write(mapped, 1, window=window)
            mapped = np.full(background.shape, -9999, dtype="float32")
            mapped[background] = total
            if not np.isfinite(mapped[background]).all() or (mapped[background] == -9999).any():
                raise ValueError("Nonfinite/NoData-colliding IV_total; inspect class statistics")
            total_out.write(mapped, 1, window=window)
            if i % 10 == 0:
                logging.info("IV surface rows %d/%d", window.row_off+window.height, reference["height"])
