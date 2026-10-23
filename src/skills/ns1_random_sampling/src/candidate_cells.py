"""Memory-bounded exact raster-center masks (no pixel polygons)."""
import logging
import numpy as np
import rasterio
import shapely
from .utils import centers, windows


def build_candidate_mask(raster, unions: dict, destination, chunk_rows: int) -> dict:
    """Write 0/1/255 mask in stripes; boundary included, exclusion edges excluded."""
    counts = dict(raster_valid_cell_count=0, nodata_cell_count=0, boundary_cell_count=0,
                  study_valid_cell_count=0, water_excluded_cell_count=0,
                  buffer_excluded_cell_count=0, candidate_cell_count=0)
    profile = dict(driver="GTiff", width=raster.width, height=raster.height, count=1,
                   crs=raster.crs, transform=raster.transform, dtype="uint8", nodata=255,
                   compress="deflate", tiled=True, blockxsize=256, blockysize=256,
                   BIGTIFF="IF_SAFER")
    with rasterio.open(destination, "w", **profile) as output:
        output.update_tags(definition="1=valid NS1 center; 0=excluded; 255=reference invalid",
                           center_rule="boundary covers; water and buffer do not intersect")
        for index, window in enumerate(windows(raster.width, raster.height, chunk_rows)):
            data = raster.read(1, window=window, masked=True)
            valid = ~np.ma.getmaskarray(data) & np.isfinite(data.data)
            rows = np.arange(window.row_off, window.row_off + window.height)[:, None]
            cols = np.arange(raster.width)[None, :]
            x, y = centers(raster.transform, rows, cols)
            boundary = shapely.intersects_xy(unions["boundary"], x, y)
            water = shapely.intersects_xy(unions["water"], x, y)
            buffer = shapely.intersects_xy(unions["historical_landslide_buffer"], x, y)
            study = valid & boundary
            candidate = study & ~water & ~buffer
            masks = [valid, ~valid, boundary, study, study & water, study & buffer, candidate]
            for key, mask in zip(counts, masks):
                counts[key] += int(np.count_nonzero(mask))
            encoded = candidate.astype("uint8")
            encoded[~valid] = 255
            output.write(encoded, 1, window=window)
            if index % 10 == 0:
                logging.info("Mask rows %d/%d", window.row_off + window.height, raster.height)
    if counts["raster_valid_cell_count"] == 0:
        raise ValueError(f"Reference raster {raster.name} has no valid cells; check NoData/mask/band 1")
    return counts
