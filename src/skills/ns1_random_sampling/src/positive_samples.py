"""One representative point per valid source landslide record."""
import numpy as np
import pandas as pd
from rasterio.transform import rowcol


def create_positive_samples(landslides, raster):
    """Preserve source attributes, renaming only collisions with generated fields."""
    result = landslides.copy()
    reserved = {"positive_id", "source_fid", "point_method", "label", "row", "col", "cell_id", "x", "y", "fid"}
    renames = {}
    for field in result.columns:
        if field.lower() in reserved:
            new = f"source_{field}"
            while new in result.columns or new in renames.values():
                new = "source_" + new
            renames[field] = new
    result = result.rename(columns=renames)
    points, methods = [], []
    for polygon in landslides.geometry:
        point = polygon.centroid
        method = "centroid"
        if not polygon.contains(point):
            point, method = polygon.representative_point(), "inside_fallback"
        if not polygon.contains(point):
            raise ValueError("Representative point is not inside its source polygon")
        points.append(point)
        methods.append(method)
    result.geometry = points
    result["source_fid"] = landslides.index.astype(str)
    result = result.reset_index(drop=True)
    result["positive_id"] = [f"NS1_P{i+1:04d}" for i in range(len(result))]
    result["point_method"] = methods
    result["label"] = 1
    result["x"], result["y"] = result.geometry.x, result.geometry.y
    rows, cols = rowcol(raster.transform, result.x, result.y)
    result["row"], result["col"] = np.asarray(rows, dtype="int64"), np.asarray(cols, dtype="int64")
    inside = (result.row >= 0) & (result.row < raster.height) & (result.col >= 0) & (result.col < raster.width)
    result["cell_id"] = pd.array([int(r)*raster.width+int(c) if ok else None
                                 for r, c, ok in zip(rows, cols, inside)], dtype="Int64")
    return result, renames
