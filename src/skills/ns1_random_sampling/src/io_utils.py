"""Read and validate only the five explicitly configured datasets."""
import logging
from pathlib import Path
import geopandas as gpd
import rasterio
import shapely
from pyproj import CRS


def polygonal(geom):
    """Keep polygonal parts of make_valid output without exploding source records."""
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if geom.geom_type == "GeometryCollection":
        parts = [polygonal(g) for g in geom.geoms]
        parts = [g for g in parts if g is not None and not g.is_empty]
        return shapely.union_all(parts) if parts else None
    return None


def clean_layer(frame: gpd.GeoDataFrame, name: str) -> tuple[gpd.GeoDataFrame, dict]:
    """Repair invalid polygons; exclude unusable positives, fail on lost mask geometry."""
    stats = dict(original_record_count=len(frame), null_geometry_count=0,
                 empty_geometry_count=0, invalid_geometry_count=0,
                 repaired_geometry_count=0, unusable_geometry_count=0, excluded_source_fids=[])
    keep, geoms = [], []
    for fid, geom in frame.geometry.items():
        result = geom
        if geom is None:
            stats["null_geometry_count"] += 1
        elif geom.is_empty:
            stats["empty_geometry_count"] += 1
            result = None
        else:
            invalid = not geom.is_valid
            stats["invalid_geometry_count"] += int(invalid)
            try:
                result = polygonal(shapely.make_valid(geom) if invalid else geom)
            except shapely.errors.GEOSException:
                result = None
            if result is None or result.is_empty or not result.is_valid or result.area <= 0:
                result = None
                stats["unusable_geometry_count"] += 1
            elif invalid:
                stats["repaired_geometry_count"] += 1
        if result is None:
            stats["excluded_source_fids"].append(str(fid))
        else:
            keep.append(fid)
            geoms.append(result)
    cleaned = frame.loc[keep].copy()
    cleaned.geometry = geoms
    stats["valid_record_count"] = len(cleaned)
    return cleaned, stats


def read_inputs(paths: dict[str, Path], encoding: str | dict | None, report: dict):
    """Validate CRS before geometry processing; never reproject or create buffers."""
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{key}: {path} does not exist; check YAML input path")
    layers = {}
    for key, path in paths.items():
        if key == "reference_raster":
            continue
        try:
            layer_encoding = encoding.get(key) if isinstance(encoding, dict) else encoding
            kwargs = {"encoding": layer_encoding} if layer_encoding else {}
            layers[key] = gpd.read_file(path, engine="pyogrio", fid_as_index=True, **kwargs)
        except Exception as exc:
            raise ValueError(f"Cannot read {key}: {path}; check shapefile sidecars/encoding: {exc}") from exc
    with rasterio.open(paths["reference_raster"]) as raster:
        report["crs"] = {key: str(layer.crs) for key, layer in layers.items()}
        report["crs"]["reference_raster"] = str(raster.crs)
        report["reference_raster"] = dict(width=raster.width, height=raster.height,
            resolution_x=raster.res[0], resolution_y=raster.res[1], extent=list(raster.bounds),
            transform=list(raster.transform), nodata=raster.nodata, dtype=raster.dtypes[0],
            band=1, crs_wkt=raster.crs.to_wkt() if raster.crs else None)
        for key, crs in report["crs"].items():
            logging.info("CRS %s (%s): %s", key, paths[key], crs)
        if raster.crs is None or any(layer.crs is None for layer in layers.values()):
            raise ValueError(f"Missing CRS; check input projection definitions: {report['crs']}")
        if any(CRS(layer.crs) != CRS(raster.crs) for layer in layers.values()):
            raise ValueError(f"CRS mismatch; align inputs explicitly before running: {report['crs']}")
        if not raster.crs.is_projected:
            raise ValueError("Reference CRS must be projected; check reference raster and vectors")
    report["geometry"] = {}
    errors = []
    for key, layer in layers.items():
        cleaned, stats = clean_layer(layer, key)
        report["geometry"][key] = stats
        logging.info("Geometry %s: %s", key, stats)
        layers[key] = cleaned
        if not len(cleaned) and key != "water":
            errors.append(f"{key}: {paths[key]} has no usable polygon records; save/export a complete existing layer")
        if key != "historical_landslides" and stats["excluded_source_fids"]:
            errors.append(f"{key}: {paths[key]} has unusable mask geometry; repair source layer explicitly")
    if errors:
        raise ValueError("; ".join(errors))
    return layers


def spatial_unions(layers: dict) -> dict:
    """Build prepared vector geometries for exact center predicates."""
    unions = {key: shapely.union_all(layer.geometry.to_numpy()) for key, layer in layers.items()}
    for geom in unions.values():
        shapely.prepare(geom)
    return unions
