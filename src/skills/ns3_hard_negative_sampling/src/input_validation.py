import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from shapely import union_all, covers
from .utils import csv
import hashlib


def read_points(path, ref, id_field=None, require_label=None):
    columns=([id_field] if id_field else [])+(['label'] if require_label is not None else [])
    points=gpd.read_file(path,columns=columns)
    if not len(points) or points.crs!=ref.crs or points.geometry.isna().any() or points.geometry.is_empty.any() or not points.geom_type.eq('Point').all():
        raise ValueError(f'Empty, invalid or CRS-mismatched point layer: {path}')
    if not points.geometry.is_valid.all():raise ValueError(f'Invalid point geometries: {path}')
    if id_field and (id_field not in points or points[id_field].isna().any() or points[id_field].duplicated().any()):
        raise ValueError(f'Missing/null/duplicate point IDs: {path}')
    if require_label is not None and ('label' not in points or not points.label.eq(require_label).all()):
        raise ValueError(f'Invalid labels in {path}')
    x,y=points.geometry.x.to_numpy(),points.geometry.y.to_numpy()
    if not (np.isfinite(x)&np.isfinite(y)).all():raise ValueError(f'Nonfinite coordinates: {path}')
    rr,cc=rasterio.transform.rowcol(ref.transform,x,y)
    points['row']=np.asarray(rr,dtype='int64');points['col']=np.asarray(cc,dtype='int64')
    if ((points.row<0)|(points.row>=ref.height)|(points.col<0)|(points.col>=ref.width)).any():raise ValueError(f'Points outside reference grid: {path}')
    return points


def validate_positive(paths,cfg,report):
    with rasterio.open(paths['reference_raster']) as ref:
        positive=read_points(paths['positive'],ref,cfg['inputs']['positive']['id_field'],1)
    digest=hashlib.sha256()
    with paths['positive'].open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    report.update(positive_count=len(positive),positive_source_path=str(paths['positive']),positive_sha256=digest.hexdigest(),
        positive_duplicate_cell_count=int(positive.duplicated(['row','col']).sum()))
    total=cfg['sampling']['total_negative_count']
    if total!='auto' and total!=len(positive):raise ValueError('Explicit total_negative_count must equal positive count')
    hard=round(len(positive)*cfg['sampling']['hard_ratio'])
    report['requested_counts']={'random':len(positive)-hard,'hard':hard,'all':len(positive)}
    return positive


def validate_spatial(paths, inventory, cfg, run, report):
    layers = {'inventory': inventory}
    records = []
    errors = []
    with rasterio.open(paths['reference_raster']) as ref, rasterio.open(paths['hardness_raster']) as score:
        signature = lambda r: (r.crs, r.shape, r.transform, r.res, r.bounds)
        if signature(ref) != signature(score) or ref.count != 1 or score.count != 1:
            errors.append('Reference/hardness grids differ or are not single-band')
        crs = CRS(ref.crs)
        if not crs.is_projected or len(crs.axis_info) < 2:
            errors.append('Projected CRS required for metric areas and distances')
            unit = 1.
        else:
            unit = crs.axis_info[0].unit_conversion_factor
            if not np.isclose(unit, crs.axis_info[1].unit_conversion_factor):
                errors.append('Different horizontal axis units are unsupported')
        t = ref.transform
        report['pixel_area_km2'] = abs(t.a*t.e-t.b*t.d)*unit**2/1e6
        report['grid'] = dict(crs=str(ref.crs), shape=list(ref.shape), transform=list(t),
            resolution=list(ref.res), reference_nodata=str(ref.nodata), hardness_nodata=str(score.nodata), unit_to_m=unit)
        for key in ('boundary','water','historical_buffer'):
            layers[key] = gpd.read_file(paths[key], columns=[])
        for key, layer in layers.items():
            bad = layer.geometry.isna() | layer.geometry.is_empty | ~layer.geometry.is_valid | ~layer.geom_type.isin(['Polygon','MultiPolygon'])
            match = layer.crs == ref.crs
            records.append(dict(layer=key,count=len(layer),crs_match=match,invalid_geometry_count=int(bad.sum())))
            if not match or bad.any() or (key != 'water' and len(layer)==0):
                errors.append(f'Invalid/empty/nonpolygon geometries or CRS mismatch: {key}')
    csv(pd.DataFrame(records), run/'qc/spatial_input_validation.csv')
    report['spatial_inputs'] = records
    if errors:
        report['spatial_alignment'] = 'FAIL'
        raise ValueError('; '.join(errors))
    report['spatial_alignment'] = 'PASS'
    # Geometric buffer footprints below are temporary QC probes only, never replacement outputs.
    buf = union_all(layers['historical_buffer'].geometry.to_numpy())
    historical = inventory[inventory.ever_landslide==1]
    dist = cfg['buffer_qc']['expected_distance_m']
    tolerance = cfg['buffer_qc']['tolerance_m']
    area_tolerance = cfg['buffer_qc']['missing_area_tolerance_m2']
    checks = []
    for row in historical.itertuples():
        covered = bool(covers(buf, row.geometry))
        probe = row.geometry.buffer((dist-tolerance)/unit)
        missing_area = float(probe.difference(buf).area*unit**2)
        checks.append(dict(feature_id=int(row.feature_id),object_covered=covered,
                           expected_buffer_missing_area_m2=missing_area,buffer_complete=covered and missing_area<=area_tolerance))
    frame = pd.DataFrame(checks,columns=['feature_id','object_covered','expected_buffer_missing_area_m2','buffer_complete'])
    csv(frame, run/'qc/historical_buffer_coverage.csv')
    report['buffer_qc'] = dict(status='PASS' if len(frame) and frame.buffer_complete.all() else 'FAIL',
        checked_objects=len(frame),uncovered_objects=int((~frame.object_covered).sum()),
        incomplete_buffer_objects=int((~frame.buffer_complete).sum()),expected_distance_m=dist,tolerance_m=tolerance,
        missing_area_tolerance_m2=area_tolerance, max_missing_area_m2=float(frame.expected_buffer_missing_area_m2.max()) if len(frame) else None,
        provisional=not report['inventory_states_valid'])
    if report['buffer_qc']['status'] != 'PASS':
        raise ValueError('Historical buffer coverage/width QC failed; see historical_buffer_coverage.csv')
    return layers
