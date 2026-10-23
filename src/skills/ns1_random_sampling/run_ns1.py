"""CLI entry point for reusable NS1 random negative sampling."""
import argparse
from datetime import datetime
import importlib.metadata
import logging
from pathlib import Path
import sys
import pandas as pd
import rasterio
import yaml
from src.config import load_config
from src.utils import create_run, setup_logging, write_json
from src.io_utils import read_inputs, spatial_unions
from src.positive_samples import create_positive_samples
from src.candidate_cells import build_candidate_mask
from src.random_sampling import sample_negative_cells
from src.qc import check_samples, write_reports


def run(config_path: Path) -> Path:
    """Execute a run; retain diagnostics and raise on any validation/write failure."""
    cfg, paths, output_root = load_config(config_path.resolve())
    seed = cfg["sampling"]["random_seed"]
    directory = create_run(output_root, seed)
    setup_logging(directory)
    report = dict(run_id=directory.name, timestamp=datetime.now().astimezone().isoformat(),
                  random_seed=seed, sample_count_mode=cfg["sampling"]["sample_count_mode"],
                  input_paths={key: str(path) for key, path in paths.items()}, qc_status="FAIL",
                  python_version=sys.version, versions={name: importlib.metadata.version(name) for name in
                  ("geopandas", "rasterio", "shapely", "numpy", "pandas", "pyyaml", "pyogrio")})
    try:
        logging.info("[1/8] Loading configuration: %s", config_path)
        (directory / "config/config_used.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        report["input_file_stats"] = {key: {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                                      for key, path in paths.items() if path.is_file()}
        logging.info("[2/8] Validating inputs")
        layers = read_inputs(paths, cfg["processing"]["vector_encoding"], report)
        unions = spatial_unions(layers)
        with rasterio.open(paths["reference_raster"]) as raster:
            logging.info("[3/8] Creating positive representative points")
            positive, renames = create_positive_samples(layers["historical_landslides"], raster)
            report["positive"] = dict(positive_sample_count=len(positive),
                centroid_count=int((positive.point_method == "centroid").sum()),
                inside_fallback_count=int((positive.point_method == "inside_fallback").sum()),
                attribute_renames=renames)
            target = len(positive) if cfg["sampling"]["sample_count_mode"] == "auto" else cfg["sampling"]["manual_sample_count"]
            report["sampling"] = dict(random_seed=seed, target_negative_count=target, replacement=False,
                                      sample_count_mode=cfg["sampling"]["sample_count_mode"])
            logging.info("[4/8] Building raster masks")
            mask_path = directory / "spatial/candidate_mask.tif"
            chunk_rows = cfg["processing"]["chunk_rows"]
            report["spatial_mask"] = build_candidate_mask(raster, unions, mask_path, chunk_rows)
            logging.info("[5/8] Building candidate cell pool (implicit row-major mask)")
            candidate_count = report["spatial_mask"]["candidate_cell_count"]
            logging.info("Candidate cells: %d", candidate_count)
            logging.info("[6/8] Random sampling")
            negative = sample_negative_cells(mask_path, candidate_count, target, seed, chunk_rows)
            report["sampling"].update(actual_negative_count=len(negative), candidate_cell_count=candidate_count)
            logging.info("[7/8] Running QC")
            report.update(check_samples(positive, negative, layers["historical_landslides"], unions, raster, mask_path, target))
            if report["qc_status"] != "PASS":
                raise ValueError(f"Conflict QC failed: {report['conflicts']}; check input geometry and masks")
            logging.info("[8/8] Writing outputs")
            positive.to_file(directory / "samples/positive_samples.gpkg", layer="positive_samples", driver="GPKG", index=False)
            negative.to_file(directory / "samples/ns1_negative_samples.gpkg", layer="ns1_negative_samples", driver="GPKG", index=False)
            common = ["sample_id", "label", "sample_type", "x", "y", "row", "col", "cell_id", "strategy"]
            pos_csv = positive.assign(sample_id=positive.positive_id, sample_type="positive", strategy="Historical")
            neg_csv = negative.assign(sample_id=negative.negative_id, sample_type="negative")
            pd.concat([pos_csv[common], neg_csv[common]], ignore_index=True).to_csv(
                directory / "samples/ns1_all_samples.csv", index=False, encoding="utf-8-sig")
            report.update(positive_count=len(positive), negative_count=len(negative), candidate_cell_count=candidate_count,
                          reference_raster_resolution=list(raster.res), CRS=str(raster.crs))
        write_reports(directory, report)
        report["output_paths"] = [str(p) for p in sorted(directory.rglob("*")) if p.is_file()] + [str(directory / "run_metadata.json")]
        write_json(directory / "run_metadata.json", report)
        logging.info("NS1 RANDOM SAMPLING COMPLETED\nPositive samples: %d\nNegative samples: %d\nCandidate cells: %d\nRandom seed: %d\nOutput directory: %s\nQC status: PASS",
                     len(positive), len(negative), candidate_count, seed, directory)
        return directory
    except Exception as exc:
        report.update(qc_status="FAIL", error=str(exc))
        logging.exception("NS1 failed. Output directory: %s. Check inputs/configuration and output write permissions.", directory)
        try:
            write_reports(directory, report)
            write_json(directory / "run_metadata.json", report)
        except Exception:
            logging.exception("Cannot write failure report under %s; check disk space and permissions", directory)
        raise


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
