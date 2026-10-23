"""Existing grid helpers reused through explicit project package imports."""
from contextlib import ExitStack
import rasterio
from skills.ns1_random_sampling.src.utils import centers, windows
from skills.lsm_modeling.src.utils import valid_values, file_snapshot


def factor_windows(paths: dict, reference_name: str, chunk_rows: int):
    """Yield raw rasters in fixed row-major stripes with bounded memory."""
    with ExitStack() as stack:
        readers = {n: stack.enter_context(rasterio.open(p)) for n, p in paths.items()}
        ref = readers[reference_name]
        for window in windows(ref.width, ref.height, chunk_rows):
            yield window, {n: r.read(1, window=window, masked=True) for n, r in readers.items()}


def profile(ref: dict, dtype: str, nodata):
    """Clean single-band output with exact reference grid."""
    return dict(driver="GTiff", width=ref["width"], height=ref["height"], transform=ref["transform"],
        crs=ref["crs"], count=1, dtype=dtype, nodata=nodata, tiled=True,
        blockxsize=256, blockysize=256, compress="deflate", BIGTIFF="IF_SAFER")
