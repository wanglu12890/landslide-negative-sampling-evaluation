"""Run configurable RF training, evaluation and full-grid susceptibility prediction."""
import argparse
from datetime import datetime
import importlib.metadata
import logging
from pathlib import Path
import sys
import yaml
from src.config import load_config
from src.output_utils import create_run, setup_logging, write_json, write_qc
from src.utils import file_snapshot
from src.raster_validation import validate_rasters
from src.feature_extraction import load_samples, extract_features
from src.dataset_builder import build_dataset, summarize_factors, split_dataset
from src.rf_model import train_rf
from src.evaluation import evaluate
from src.spatial_prediction import predict_area, validate_and_summarize


def run(config_path: Path) -> Path:
    """Keep all experiment settings and diagnostics in an independent run."""
    cfg, factors, samples, output_root = load_config(config_path.resolve())
    directory = create_run(output_root, cfg["split"]["random_seed"])
    setup_logging(directory)
    report = dict(run_id=directory.name, timestamp=datetime.now().astimezone().isoformat(), qc_status="FAIL",
        experiment_name=cfg["experiment_name"], input_paths={k: str(v) for k, v in {**factors, **samples}.items()},
        python_version=sys.version, versions={name: importlib.metadata.version(name) for name in
        ("geopandas", "rasterio", "numpy", "pandas", "scikit-learn", "shapely", "joblib", "matplotlib", "pyyaml")})
    try:
        logging.info("[1/10] Loading configuration: %s", config_path)
        (directory / "config/config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        initial = file_snapshot({**factors, **samples})
        report["input_file_stats"] = initial
        logging.info("[2/10] Validating factor rasters")
        reference = validate_rasters(factors, cfg["reference_raster"], directory / "qc/raster_alignment_report.csv", report)
        logging.info("[3/10] Loading samples")
        points = load_samples(samples, reference, cfg["allow_imbalanced_samples"], report)
        logging.info("[4/10] Extracting factor values")
        extracted, bad = extract_features(points, factors, directory / "data/invalid_samples.csv", report)
        logging.info("[5/10] Building modeling dataset")
        prefix = cfg["experiment_name"].removesuffix("_RF").lower()
        data = build_dataset(extracted, bad, cfg, directory / f"data/{prefix}_modeling_dataset.csv", report)
        summarize_factors(data, factors, cfg, directory, report)
        logging.info("[6/10] Splitting train/test data")
        train, test = split_dataset(data, cfg, directory, report)
        logging.info("[7/10] Training Random Forest")
        model = train_rf(train, test, cfg, directory, report)
        logging.info("[8/10] Evaluating model")
        report["metrics"] = evaluate(model, test, cfg["factors"]["feature_order"], directory)
        logging.info("Test ROC-AUC=%.6f, F1=%.6f", report["metrics"]["ROC_AUC"], report["metrics"]["F1"])
        logging.info("[9/10] Predicting full study area")
        output_raster = directory / f"spatial/{cfg['experiment_name']}_susceptibility.tif"
        predict_area(model, factors, cfg["reference_raster"], output_raster, cfg["prediction"]["batch_size"])
        logging.info("[10/10] Writing outputs and QC report; validating raster read-back")
        checks, stats = validate_and_summarize(output_raster, factors, cfg["reference_raster"],
            cfg["prediction"]["batch_size"], directory / "spatial/susceptibility_statistics.csv")
        checks["input_files_unchanged"] = initial == file_snapshot({**factors, **samples})
        report.update(output_validation=checks, prediction={**stats, "output_raster": str(output_raster)})
        if not all(checks.values()):
            raise ValueError("Output validation failed or input files changed during this run; inspect run metadata")
        report["qc_status"] = "PASS"
        write_qc(directory, report)
        report["output_paths"] = [str(path) for path in sorted(directory.rglob("*")) if path.is_file()] + [str(directory / "run_metadata.json")]
        write_json(directory / "run_metadata.json", report)
        logging.info("LSM RANDOM FOREST COMPLETED\nPositive samples: %d\nNegative samples: %d\nTraining samples: %d\nTest samples: %d\nROC-AUC: %.6f\nF1: %.6f\nValid prediction cells: %d\nOutput raster: %s\nQC status: PASS\nOutput directory: %s",
            report["samples"]["positive"], report["samples"]["negative"],
            len(train), len(test), report["metrics"]["ROC_AUC"], report["metrics"]["F1"], stats["valid_cell_count"], output_raster, directory)
        return directory
    except Exception as exc:
        report.update(qc_status="FAIL", error=str(exc))
        logging.exception("LSM failed: %s. Diagnostics: %s", exc, directory)
        try:
            write_qc(directory, report)
            write_json(directory / "run_metadata.json", report)
        except Exception:
            logging.exception("Failed to write diagnostics; check output permissions and disk space: %s", directory)
        raise
    finally:
        for handler in logging.getLogger().handlers[:]:
            handler.close()
            logging.getLogger().removeHandler(handler)


def main() -> int:
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
