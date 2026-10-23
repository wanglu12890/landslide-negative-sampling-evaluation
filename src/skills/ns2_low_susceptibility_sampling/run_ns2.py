"""Execute fixed-classification IV screening and NS2 sampling, without RF training."""
import argparse
from datetime import datetime
import importlib.metadata
import logging
from pathlib import Path
import shutil
import sys
import pandas as pd
import geopandas as gpd
import yaml

# Explicit project-package imports avoid collisions between the existing src packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from skills.ns2_low_susceptibility_sampling.src.config import load_config
from skills.ns2_low_susceptibility_sampling.src.output_utils import create_run, setup_logging, write_json, write_csv, write_report
from skills.ns2_low_susceptibility_sampling.src.raster_validation import validate_rasters
from skills.ns2_low_susceptibility_sampling.src.sample_validation import load_vectors, load_positive
from skills.ns2_low_susceptibility_sampling.src.candidate_mask import build_background, build_low_mask
from skills.ns2_low_susceptibility_sampling.src.iv_calculation import calculate_iv, build_iv_surface
from skills.ns2_low_susceptibility_sampling.src.sampling import sample_negatives
from skills.ns2_low_susceptibility_sampling.src.qc import check_samples, validate_outputs
from skills.ns2_low_susceptibility_sampling.src.utils import file_snapshot


def run(config_path: Path) -> Path:
    """Persist configuration/provenance and fail visibly without changing source data."""
    cfg, factors, vectors, positive_path, comparison, root = load_config(config_path.resolve())
    seed = cfg["sampling"]["random_seed"]
    directory = create_run(root, seed)
    setup_logging(directory)
    inputs = {**factors, **vectors, "positive_samples": positive_path}
    if comparison:
        inputs["comparison_ns1_negative"] = comparison
    report = dict(run_id=directory.name, timestamp=datetime.now().astimezone().isoformat(), qc_status="FAIL",
        input_paths={k: str(v) for k, v in inputs.items()},
        configuration=dict(feature_order=list(factors), factor_count=len(factors), continuous=cfg["factors"]["continuous"],
            categorical=cfg["factors"]["categorical"], continuous_count=len(cfg["factors"]["continuous"]),
            categorical_count=len(cfg["factors"]["categorical"]), random_seed=seed,
            quantile=cfg["low_susceptibility"]["quantile"], smoothing=cfg["information_value"]["smoothing"]),
        versions={n: importlib.metadata.version(n) for n in ("numpy", "pandas", "geopandas", "rasterio", "shapely", "pyyaml")})
    try:
        logging.info("[1/11] Loading configuration: %s", config_path)
        (directory / "config/config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        logging.info("[2/11] Validating configuration: %d enabled factors", len(factors))
        for key, path in inputs.items():
            if not path.is_file():
                raise FileNotFoundError(f"{key}: {path}; check configured input path")
        snapshot = file_snapshot(inputs)
        report["input_file_stats"] = snapshot
        logging.info("[3/11] Validating factor rasters")
        reference = validate_rasters(factors, cfg["reference_raster"], directory / "qc/raster_alignment_report.csv", report)
        unions = load_vectors(vectors, reference, report)
        logging.info("[4/11] Loading positive samples")
        original, points, positives = load_positive(positive_path, reference, factors, unions, cfg, directory, report)
        logging.info("[5/11] Building IV background domain")
        logging.info("[6/11] Reclassifying factors using configured breaks")
        counts, observed = build_background(factors, reference, unions, cfg, directory, report)
        logging.info("[7/11] Calculating information values")
        lookup, table = calculate_iv(counts, observed, positives, cfg, reference, directory, report)
        logging.info("[8/11] Building IV_total surface")
        build_iv_surface(factors, reference, unions, cfg, lookup, directory)
        logging.info("[9/11] Building low-susceptibility candidate pool")
        threshold, pool = build_low_mask(reference, cfg, directory, report)
        logging.info("[10/11] Sampling NS2 negatives")
        required = len(original) if cfg["sampling"]["sample_count_mode"] == "auto" else cfg["sampling"]["manual_sample_count"]
        negatives = sample_negatives(directory, required, pool, threshold, cfg)
        checks, overlap = check_samples(negatives, original, factors, reference, unions, threshold, required, comparison, directory)
        report["conflicts"] = checks
        report["sampling"] = dict(required_negatives=required, actual_negatives=len(negatives), positive_count=len(original),
            ratio=f"{len(original)}:{len(negatives)}", sample_count_mode=cfg["sampling"]["sample_count_mode"],
            random_seed=seed, replacement=False, **overlap)
        if any(checks.values()):
            raise ValueError(f"Negative sample QC failed: {checks}")
        logging.info("[11/11] Writing outputs and QC report; checking every raster cell")
        report["output_validation"] = validate_outputs(factors, reference, unions, cfg, lookup, threshold, directory)
        if positive_path.suffix.lower() == ".gpkg":
            shutil.copy2(positive_path, directory / "samples/positive_samples.gpkg")
        else:
            original.to_file(directory / "samples/positive_samples.gpkg", driver="GPKG", index=False)
        negatives.to_file(directory / "samples/ns2_negative_samples.gpkg", layer="ns2_negative_samples", driver="GPKG", index=False)
        copied = gpd.read_file(directory / "samples/positive_samples.gpkg")
        if not original.equals(copied):
            raise ValueError("Copied positives differ from original; original positive set must remain unchanged")
        points = points.assign(sample_type="positive", strategy="Historical")
        neg_csv = negatives.assign(sample_id=negatives.negative_id, sample_type="negative")
        columns = ["sample_id", "label", "sample_type", "x", "y", "row", "col", "strategy"]
        write_csv(pd.concat([points[columns], neg_csv[columns]], ignore_index=True), directory / "samples/ns2_all_samples.csv")
        write_csv(pd.DataFrame([report["sampling"]]), directory / "qc/sample_statistics.csv")
        report["output_validation"].update(positive_samples_unchanged=True, input_files_unchanged=snapshot == file_snapshot(inputs))
        if not report["output_validation"]["input_files_unchanged"]:
            raise ValueError("Input files changed during the run; repeat with stable inputs")
        report["qc_status"] = "PASS"
        write_report(directory, report)
        report["output_paths"] = [str(p) for p in sorted(directory.rglob("*")) if p.is_file()] + [str(directory / "run_metadata.json")]
        write_json(directory / "run_metadata.json", report)
        logging.info("NS2 LOW-SUSCEPTIBILITY SAMPLING COMPLETED\nFactors: %d\nPositive samples: %d\nBase candidate cells: %d\nLow candidate cells: %d\nIV quantile threshold: %.9f\nNegative samples: %d\nP:N: %s\nRandom seed: %d\nQC status: PASS\nOutput directory: %s",
            len(factors), len(original), report["domains"]["base_candidate_cells"], pool, threshold, len(negatives), report["sampling"]["ratio"], seed, directory)
        return directory
    except Exception as exc:
        report.update(qc_status="FAIL", error=str(exc))
        logging.exception("NS2 failed. Diagnostics: %s", directory)
        try:
            write_report(directory, report)
            write_json(directory / "run_metadata.json", report)
        except Exception:
            logging.exception("Cannot write failure diagnostics; check output permissions/disk space")
        raise
    finally:
        for handler in logging.getLogger().handlers[:]:
            handler.close()
            logging.getLogger().removeHandler(handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    try:
        run(args.config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
