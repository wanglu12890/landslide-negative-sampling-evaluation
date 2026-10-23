"""Run directories, logging, JSON and raster window helpers."""
from datetime import datetime
from pathlib import Path
import json
import logging
import numpy as np
from rasterio.windows import Window


def create_run(root: Path, seed: int) -> Path:
    """Create a unique directory atomically, never overwrite a prior run."""
    stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_seed{seed}"
    run = root / stem
    run.mkdir(parents=True, exist_ok=False)
    for name in ("config", "samples", "spatial", "qc", "logs"):
        (run / name).mkdir()
    return run


def setup_logging(run: Path) -> None:
    """Log to UTF-8 file and terminal."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(run / "logs/run.log", encoding="utf-8"),
                                  logging.StreamHandler()], force=True)


def write_json(path: Path, value: dict) -> None:
    """Write portable metadata; stringify non-finite raster metadata explicitly."""
    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj):
            return str(obj)
        return obj
    path.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def windows(width: int, height: int, chunk_rows: int):
    """Yield full-width stripes in deterministic row-major order."""
    for row in range(0, height, chunk_rows):
        yield Window(0, row, width, min(chunk_rows, height - row))


def centers(transform, rows, cols):
    """Compute actual pixel centers, including rotated affine grids."""
    x = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    y = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    return np.broadcast_arrays(x, y)
