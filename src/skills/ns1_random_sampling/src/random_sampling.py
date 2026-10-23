"""Uniform sampling from an implicit, row-major candidate cell pool."""
import geopandas as gpd
import numpy as np
import rasterio
from .utils import centers, windows


def sample_negative_cells(mask_path, candidate_count: int, count: int, seed: int, chunk_rows: int):
    """Choose unique candidate ordinals, resolve row/col in bounded memory."""
    if candidate_count < count:
        raise ValueError(f"Candidate mask {mask_path}: {candidate_count} cells < target {count}; check masks/count")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidate_count, size=count, replace=False)
    order = np.argsort(chosen)
    sorted_chosen = chosen[order]
    rows, cols = np.empty(count, dtype="int64"), np.empty(count, dtype="int64")
    offset, found = 0, 0
    with rasterio.open(mask_path) as mask:
        for window in windows(mask.width, mask.height, chunk_rows):
            flat = np.flatnonzero(mask.read(1, window=window) == 1)
            lo, hi = np.searchsorted(sorted_chosen, [offset, offset + len(flat)])
            local = flat[sorted_chosen[lo:hi] - offset]
            rows[order[lo:hi]] = local // mask.width + int(window.row_off)
            cols[order[lo:hi]] = local % mask.width
            found += hi - lo
            offset += len(flat)
        if found != count or offset != candidate_count:
            raise ValueError(f"Candidate pool changed or sample count insufficient: {mask_path}")
        x, y = centers(mask.transform, rows, cols)
        return gpd.GeoDataFrame(dict(negative_id=[f"NS1_N{i+1:04d}" for i in range(count)],
            label=0, row=rows, col=cols, cell_id=rows*mask.width+cols, x=x, y=y,
            random_seed=seed, strategy="NS1_Random"), geometry=gpd.points_from_xy(x, y), crs=mask.crs)
