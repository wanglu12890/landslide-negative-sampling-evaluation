"""Shared validity, deterministic windows and provenance helpers."""
from pathlib import Path
import numpy as np
from rasterio.windows import Window


def windows(width: int, height: int, max_cells: int):
    """Yield non-overlapping row-major windows no larger than max_cells."""
    tile_width = min(width, max_cells)
    tile_height = max(1, max_cells // tile_width)
    for row in range(0, height, tile_height):
        for col in range(0, width, tile_width):
            yield Window(col, row, min(tile_width, width-col), min(tile_height, height-row))


def valid_values(array) -> np.ndarray:
    """NoData, internal raster masks, NaN and Inf all exclude a value."""
    return ~np.ma.getmaskarray(array) & np.isfinite(array.data)


def file_snapshot(paths: dict[str, Path]) -> dict:
    """Record input size and nanosecond timestamps without modifying any source."""
    return {key: dict(path=str(path), size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
            for key, path in paths.items() if path.is_file()}


def class_counts(frame) -> dict:
    """Return JSON-safe binary sample counts."""
    return {"positive": int((frame.label == 1).sum()), "negative": int((frame.label == 0).sum())}
