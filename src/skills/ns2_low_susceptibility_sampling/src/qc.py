"""Independent point checks and complete raster read-back validation."""
from contextlib import ExitStack
import geopandas as gpd
import numpy as np
import rasterio
import shapely
from rasterio.transform import rowcol
from rasterio.windows import Window
from .utils import centers, factor_windows, valid_values
from .candidate_mask import domains
from .iv_calculation import map_iv


def check_samples(negative, original, factors, reference, unions, threshold, required, comparison, run):
    """Recheck selected geometry, reference centers and each raw factor pixel."""
    geoms = negative.geometry.to_numpy()
    x, y = centers(reference["transform"], negative.row.to_numpy(), negative.col.to_numpy())
    checks = dict(negative_in_water=int(shapely.intersects(unions["water"], geoms).sum()),
        negative_in_buffer=int(shapely.intersects(unions["historical_landslide_buffer"], geoms).sum()),
        negative_outside_boundary=int((~shapely.covers(unions["boundary"], geoms)).sum()),
        duplicate_negative_cells=int(negative.cell_id.duplicated().sum()),
        duplicate_negative_points=int(negative.geometry.to_wkb().duplicated().sum()),
        count_error=abs(len(negative)-required), negative_on_factor_nodata=0,
        negative_not_low_candidate=0, negative_iv_mismatch=0,
        negative_above_threshold=int((negative.iv_total > threshold).sum()),
        negative_not_cell_center=int(((negative.geometry.x != x) | (negative.geometry.y != y)).sum()),
        invalid_cell_id=int((negative.cell_id != negative.row*reference["width"]+negative.col).sum()),
        positive_negative_coordinate_overlap=len(set(original.geometry.to_wkb()) & set(negative.geometry.to_wkb())))
    with ExitStack() as stack:
        readers = [stack.enter_context(rasterio.open(p)) for p in factors.values()]
        mask = stack.enter_context(rasterio.open(run / "spatial/low_susceptibility_mask.tif"))
        iv = stack.enter_context(rasterio.open(run / "spatial/IV_total.tif"))
        for p in negative.itertuples():
            w = Window(int(p.col), int(p.row), 1, 1)
            checks["negative_on_factor_nodata"] += int(any(not valid_values(r.read(1, window=w, masked=True)).all() for r in readers))
            checks["negative_not_low_candidate"] += int(mask.read(1, window=w)[0, 0] != 1)
            checks["negative_iv_mismatch"] += int(iv.read(1, window=w)[0, 0] != p.iv_total)
    overlap = {}
    if comparison:
        other = gpd.read_file(comparison)
        if other.crs != reference["crs"] or not other.geom_type.eq("Point").all():
            raise ValueError(f"Comparison samples have invalid CRS/geometry: {comparison}")
        rows, cols = rowcol(reference["transform"], other.geometry.x, other.geometry.y)
        cells = set(zip(rows, cols))
        count = len(cells & set(zip(negative.row, negative.col)))
        overlap = dict(overlap_count=count, overlap_ratio=count/len(negative), denominator="NS2 negative count", diagnostic_only=True)
    return checks, overlap


def validate_outputs(paths, reference, unions, cfg, lookup, threshold, run):
    """Compare every output cell to raw input validity, domains and mapped IV values."""
    count = 0
    with ExitStack() as stack:
        outputs = {key: stack.enter_context(rasterio.open(run / path)) for key, path in
            {"base": "spatial/base_candidate_mask.tif", "low": "spatial/low_susceptibility_mask.tif", "iv": "spatial/IV_total.tif"}.items()}
        factor_outputs = {n: stack.enter_context(rasterio.open(run / f"spatial/iv_factors/IV_{n}.tif")) for n in paths} if cfg["processing"]["write_iv_factors"] else {}
        for raster in [*outputs.values(), *factor_outputs.values()]:
            is_mask = raster is outputs["base"] or raster is outputs["low"]
            dtype, nodata = ("uint8", 255) if is_mask else ("float32", -9999)
            if (raster.crs != reference["crs"] or raster.transform != reference["transform"]
                or raster.width != reference["width"] or raster.height != reference["height"]
                or raster.dtypes != (dtype,) or raster.nodata != nodata):
                raise ValueError(f"Output grid/dtype/NoData mismatch: {raster.name}")
        for w, arrays in factor_windows(paths, cfg["reference_raster"], cfg["processing"]["chunk_rows"]):
            valid, background, base = domains(arrays, w, reference, unions)
            actual = outputs["iv"].read(1, window=w)
            expected_base = base.astype("uint8")
            expected_base[~valid] = 255
            expected_low = (base & (actual.astype("float64") <= threshold)).astype("uint8")
            expected_low[~valid] = 255
            if not np.array_equal(outputs["base"].read(1, window=w), expected_base) or not np.array_equal(outputs["low"].read(1, window=w), expected_low):
                raise ValueError(f"Output candidate mask does not match raw input domain at {w}")
            total = np.zeros(int(background.sum()), dtype="float64")
            for name, array in arrays.items():
                mapped = map_iv(array.data[background], name, cfg, lookup)
                total += mapped
                if name in factor_outputs:
                    factor_output = factor_outputs[name].read(1, window=w)
                    if not (np.array_equal(factor_output[background], mapped.astype("float32")) and (factor_output[~background] == -9999).all()):
                        raise ValueError(f"IV factor output mismatch: {name}, {w}")
            if not (np.array_equal(actual[background], total.astype("float32")) and (actual[~background] == -9999).all()):
                raise ValueError(f"IV_total mismatch at {w}")
            count += int(background.sum())
    return dict(all_rasters_aligned=True, masks_match_raw_inputs=True, iv_mapping_and_sum_match=True,
                iv_background_valid_cells=count, all_output_cells_checked=True)
