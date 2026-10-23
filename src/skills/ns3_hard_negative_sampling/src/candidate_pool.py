from contextlib import ExitStack
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from .utils import windows, valid, profile, csv


def polygon_mask(layer, window, ref):
    shape = (int(window.height),int(window.width))
    transform = ref.window_transform(window)
    corners = [transform*(c,r) for c,r in ((0,0),(window.width,0),(0,window.height),(window.width,window.height))]
    xs,ys = zip(*corners)
    indices = layer.sindex.query(box(min(xs),min(ys),max(xs),max(ys)), predicate='intersects')
    if not len(indices):
        return np.zeros(shape,dtype=bool)
    return rasterize(((g,1) for g in layer.geometry.iloc[indices]), out_shape=shape,
        transform=ref.window_transform(window),fill=0,all_touched=False,dtype='uint8').astype(bool)


def build_candidates(paths, layers, cfg, run, report):
    counts = {k:0 for k in ['candidate']+cfg['exclusion']['priority']}
    chunks = []; valid_study = 0; total = 0
    with ExitStack() as stack:
        ref = stack.enter_context(rasterio.open(paths['reference_raster']))
        score = stack.enter_context(rasterio.open(paths['hardness_raster']))
        mask = stack.enter_context(rasterio.open(run/'spatial/ns3_candidate_mask.tif','w',**profile(ref)))
        reason = stack.enter_context(rasterio.open(run/'spatial/ns3_exclusion_reason.tif','w',**profile(ref)))
        mask.update_tags(interpretation='Exploration pool, not confirmed stable negatives')
        reason.update_tags(**{f'code_{v}':k for k,v in cfg['exclusion']['codes'].items()},code_0='candidate')
        for w in windows(ref,cfg['processing']['window_rows']):
            a = ref.read(1,window=w,masked=True); s=score.read(1,window=w,masked=True)
            sv = valid(s)
            if ((s.data[sv]<0)|(s.data[sv]>1)).any():
                raise ValueError(f'Hardness outside [0,1] at {w}; no clipping allowed')
            boundary = polygon_mask(layers['boundary'],w,ref)
            valid_study += int((boundary & valid(a)).sum())
            conditions = dict(outside_boundary=~boundary,reference_nodata=~valid(a),
                water=polygon_mask(layers['water'],w,ref), historical_landslide_buffer=polygon_mask(layers['historical_buffer'],w,ref),hardness_nodata=~sv)
            codes = np.zeros(a.shape,dtype='uint8')
            for key in cfg['exclusion']['priority']:
                select = (codes==0)&conditions[key]
                codes[select]=cfg['exclusion']['codes'][key]
                counts[key]+=int(select.sum())
            candidate=codes==0
            counts['candidate']+=int(candidate.sum()); total+=codes.size
            chunks.append(s.data[candidate])
            mask.write(candidate.astype('uint8'),1,window=w); reason.write(codes,1,window=w)
    area = report['pixel_area_km2']
    report['candidate_cell_count']=counts['candidate']; report['valid_study_area_cell_count']=valid_study
    report['candidate_area_km2']=counts['candidate']*area
    report['candidate_area_ratio_of_valid_study_area']=counts['candidate']/valid_study if valid_study else None
    report['hardness_valid_range']='PASS: all valid input raster scores checked'
    summary=pd.DataFrame([dict(reason=k,cell_count=n,area_km2=n*area,area_ratio=n/total,
        area_ratio_denominator='full_reference_grid',ratio_of_valid_study_area=n/valid_study if k=='candidate' and valid_study else np.nan) for k,n in counts.items()])
    csv(summary,run/'statistics/exclusion_summary.csv')
    if not counts['candidate']:
        raise ValueError('Candidate pool is empty')
    if sum(counts.values())!=total:
        raise ValueError('Exclusion partition failed')
    report['exclusion_partition']='PASS'
    return np.concatenate(chunks)
