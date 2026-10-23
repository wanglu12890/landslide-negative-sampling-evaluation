"""Strict raster-grid compatibility checks without resampling."""
import logging
import rasterio
import pandas as pd
from .output_utils import write_csv


def validate_rasters(paths: dict, reference: str, report_path, report: dict) -> dict:
    """Audit all factors and write the alignment table before raising an error."""
    rows, metadata, errors = [], {}, []
    for factor, path in paths.items():
        try:
            with rasterio.open(path) as raster:
                raster.read(1, window=((0, 1), (0, 1)))
                metadata[factor] = dict(crs=raster.crs, width=raster.width, height=raster.height,
                    transform=raster.transform, bounds=raster.bounds, resolution=raster.res,
                    count=raster.count, dtype=raster.dtypes[0], nodata=raster.nodata)
        except Exception as exc:
            errors.append(f"{factor}: {path}: cannot read raster; check file and permissions: {exc}")
    ref = metadata.get(reference)
    for factor, path in paths.items():
        meta = metadata.get(factor)
        row = dict(factor=factor, path=str(path), alignment_status="FAIL")
        if meta:
            row.update(crs=str(meta["crs"]), width=meta["width"], height=meta["height"],
                res_x=meta["resolution"][0], res_y=meta["resolution"][1],
                xmin=meta["bounds"].left, ymin=meta["bounds"].bottom,
                xmax=meta["bounds"].right, ymax=meta["bounds"].top,
                nodata=meta["nodata"], dtype=meta["dtype"], raster_count=meta["count"], transform=str(meta["transform"]))
        if meta and ref:
            checks = dict(crs_match=meta["crs"] is not None and meta["crs"] == ref["crs"],
                shape_match=(meta["width"], meta["height"]) == (ref["width"], ref["height"]),
                transform_match=meta["transform"] == ref["transform"], bounds_match=meta["bounds"] == ref["bounds"],
                resolution_match=meta["resolution"] == ref["resolution"], single_band=meta["count"] == 1)
            row.update(checks)
            failed = [name for name, passed in checks.items() if not passed]
            row["alignment_status"] = "FAIL" if failed else "PASS"
            if failed:
                errors.append(f"{factor}: {path} differs from {reference} ({paths[reference]}): {', '.join(failed)}")
        rows.append(row)
        logging.info("Raster %s: %s", factor, row["alignment_status"])
    write_csv(pd.DataFrame(rows), report_path)
    report["rasters"] = dict(factor_names=list(paths), factor_count=len(paths), reference=reference,
                              alignment_result="FAIL" if errors else "PASS")
    if ref:
        report["rasters"].update(crs=str(ref["crs"]), resolution=list(ref["resolution"]),
            width=ref["width"], height=ref["height"], transform=list(ref["transform"]), bounds=list(ref["bounds"]))
    if errors:
        raise ValueError("Raster alignment validation failed; no resampling performed:\n" + "\n".join(errors))
    return ref
