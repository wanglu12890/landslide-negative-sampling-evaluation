"""Independent geometry and raster checks plus human-readable reports."""
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.windows import Window
from .utils import centers


def check_samples(positive, negative, landslides, unions, raster, mask_path, target: int) -> dict:
    """Test final points against original vector constraints and reference pixels."""
    points = negative.geometry.to_numpy()
    rows, cols = negative.row.to_numpy(), negative.col.to_numpy()
    checks = {
        "negative_outside_boundary": int(np.count_nonzero(~shapely.covers(unions["boundary"], points))),
        "negative_in_water": int(np.count_nonzero(shapely.intersects(unions["water"], points))),
        "negative_in_landslide_buffer": int(np.count_nonzero(shapely.intersects(unions["historical_landslide_buffer"], points))),
        "negative_in_historical_landslides": int(np.count_nonzero(shapely.intersects(unions["historical_landslides"], points))),
        "duplicate_negative_cells": int(negative.cell_id.duplicated().sum()),
        "duplicate_negative_points": int(negative.geometry.to_wkb().duplicated().sum()),
        "negative_count_error": abs(len(negative)-target),
        "positive_outside_source_polygon": int(np.count_nonzero(~shapely.contains(landslides.geometry.to_numpy(), positive.geometry.to_numpy()))),
        "negative_on_nodata": 0, "negative_outside_raster": 0, "negative_not_candidate": 0,
    }
    x, y = centers(raster.transform, rows, cols)
    checks["negative_not_cell_center"] = int(np.count_nonzero((x != negative.geometry.x) | (y != negative.geometry.y)))
    checks["negative_invalid_cell_id"] = int(np.count_nonzero(negative.cell_id != rows*raster.width+cols))
    with rasterio.open(mask_path) as mask:
        checks["mask_alignment_error"] = int((mask.crs, mask.transform, mask.width, mask.height, mask.dtypes, mask.nodata) !=
            (raster.crs, raster.transform, raster.width, raster.height, ("uint8",), 255))
        for row, col in zip(rows, cols):
            if row < 0 or col < 0 or row >= raster.height or col >= raster.width:
                checks["negative_outside_raster"] += 1
                continue
            window = Window(int(col), int(row), 1, 1)
            value = raster.read(1, window=window, masked=True)
            checks["negative_on_nodata"] += int(np.ma.getmaskarray(value).any() or not np.isfinite(value.data).all())
            checks["negative_not_candidate"] += int(mask.read(1, window=window)[0, 0] != 1)
    duplicates = positive.cell_id.dropna().value_counts()
    duplicated = duplicates[duplicates > 1]
    return dict(conflicts=checks, duplicate_positive_cell_count=len(duplicated),
                affected_positive_sample_count=int(duplicated.sum()),
                positive_outside_raster_count=int(positive.cell_id.isna().sum()),
                qc_status="PASS" if not any(checks.values()) else "FAIL")


def write_reports(run, report: dict) -> None:
    """Write reports on both success and failure; overlapping mask counts are explicit."""
    sections = [("1. Input", report.get("input_paths", {})), ("2. CRS", report.get("crs", {})),
                ("3. Reference raster", report.get("reference_raster", {})),
                ("4. Positive samples and geometry", {"positive": report.get("positive", {}), "geometry": report.get("geometry", {}),
                  "duplicate_positive_cell_count": report.get("duplicate_positive_cell_count"),
                  "affected_positive_sample_count": report.get("affected_positive_sample_count"),
                  "positive_outside_raster_count": report.get("positive_outside_raster_count")}),
                ("5. Spatial mask", report.get("spatial_mask", {})),
                ("6. Random sampling", report.get("sampling", {})),
                ("7. Conflict QC", report.get("conflicts", {}))]
    lines = ["# NS1 Random Negative Sampling QC Report", ""]
    for heading, values in sections:
        lines.extend([f"## {heading}", ""])
        lines.extend(f"- {key}: {value}" for key, value in values.items())
        lines.append("")
    lines.extend(["Mask counts overlap: water/buffer counts each refer to valid cells inside boundary; do not add them.",
                  "Candidate centers include the boundary edge and exclude water/buffer edges.",
                  "Positive duplicate cells count distinct cells with >1 positive; affected count includes all positives in those cells.",
                  "", "## 8. Final conclusion", "", f"QC Status: {report.get('qc_status', 'FAIL')}", "",
                  "NS1 random negative sampling completed successfully." if report.get("qc_status") == "PASS" else
                  f"Run failed: {report.get('error', 'Conflict QC failed; inspect checks above.')}"])
    (run / "qc/qc_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = dict(report.get("conflicts", {}))
    for key in ("duplicate_positive_cell_count", "affected_positive_sample_count", "positive_outside_raster_count", "qc_status", "error"):
        if key in report:
            summary[key] = report[key]
    pd.DataFrame(list(summary.items()), columns=["metric", "value"]).to_csv(run / "qc/qc_summary.csv", index=False, encoding="utf-8-sig")
    statistics = {**report.get("positive", {}), **report.get("sampling", {}), **report.get("spatial_mask", {})}
    pd.DataFrame(list(statistics.items()), columns=["metric", "value"]).to_csv(run / "qc/sample_statistics.csv", index=False, encoding="utf-8-sig")
