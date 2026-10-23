import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from .utils import windows,valid,qname,csv
from .candidate_pool import polygon_mask


def select_samples(paths,cfg,run,report,lo,hi):
    rng=np.random.default_rng(cfg['sampling']['seed'])
    targets={k:np.sort(rng.choice(report['pool_counts'][k],report['requested_counts'][k],replace=False)) for k in ('random','hard')}
    seen=dict(random=0,hard=0);records=[]
    low_q=cfg['hard_negative']['lower_quantile']; high_q=cfg['hard_negative']['upper_quantile']
    with rasterio.open(paths['hardness_raster']) as score,rasterio.open(run/'spatial/ns3_candidate_mask.tif') as mask:
        t=score.transform
        for w in windows(score,cfg['processing']['window_rows']):
            v=score.read(1,window=w).astype('float64');candidate=mask.read(1,window=w)==1
            conditions={'random':candidate&(v<lo),'hard':candidate&(v>=lo)&(v<hi)}
            for key,condition in conditions.items():
                positions=np.flatnonzero(condition.ravel());count=len(positions)
                left,right=np.searchsorted(targets[key],[seen[key],seen[key]+count])
                picks=positions[targets[key][left:right]-seen[key]]
                for local in picks:
                    r,c=divmod(int(local),score.width);row=r+int(w.row_off)
                    x,y=t*(c+.5,row+.5)
                    records.append(dict(row=row,col=c,x=x,y=y,label=0,negative_source=key,hardness_score=float(v[r,c]),
                        hardness_group=f'random_lt_{qname(low_q)}' if key=='random' else f'hard_{qname(low_q)}_{qname(high_q)}',
                        hardness_quantile_lower=np.nan if key=='random' else low_q,hardness_quantile_upper=low_q if key=='random' else high_q))
                seen[key]+=count
        crs=score.crs
    if any(seen[k]!=report['pool_counts'][k] for k in seen):raise ValueError('Streaming pool counts differ from full-score counts')
    frame=pd.DataFrame(records).sort_values(['negative_source','row','col'],ascending=[False,True,True],kind='stable').reset_index(drop=True)
    ordinal=frame.groupby('negative_source',sort=False).cumcount()+1
    frame.insert(0,'negative_id',[f"{cfg['sampling']['id_prefix']}_{'R' if k=='random' else 'H'}_{i:04d}" for k,i in zip(frame.negative_source,ordinal)])
    return gpd.GeoDataFrame(frame,geometry=gpd.points_from_xy(frame.x,frame.y),crs=crs)


def validate_selection(points,positive,paths,layers,cfg,run,report,lo,hi):
    duplicates=int(points.duplicated(['row','col']).sum())
    pos=set(zip(positive.row,positive.col));conflicts=sum((r,c) in pos for r,c in zip(points.row,points.col))
    counts=points.negative_source.value_counts()
    valid_counts=len(points)==len(positive) and all(int(counts.get(k,0))==report['requested_counts'][k] for k in ('random','hard'))
    random=points[points.negative_source=='random'];hard=points[points.negative_source=='hard']
    conditions={'negative_count_matches_positive':len(points)==len(positive),'source_counts_correct':valid_counts,
        'duplicate_row_col_zero':duplicates==0,'positive_conflict_zero':conflicts==0,
        'negative_ids_unique':points.negative_id.is_unique and points.negative_id.notna().all(),
        'source_complete':points.negative_source.isin(['random','hard']).all(),
        'score_complete':np.isfinite(points.hardness_score).all(),
        'random_score_valid':(random.hardness_score<lo).all(),
        'hard_score_valid':((hard.hardness_score>=lo)&(hard.hardness_score<hi)).all(),
        'no_extreme_selected':(points.hardness_score<hi).all(), 'labels_zero':points.label.eq(0).all()}
    report['duplicate_cell_count']=duplicates;report['positive_same_cell_conflict_count']=int(conflicts)
    # Independent window-wise reconstruction from raw inputs, also validates mask serialization.
    checked=0
    with rasterio.open(paths['reference_raster']) as ref,rasterio.open(paths['hardness_raster']) as score,rasterio.open(run/'spatial/ns3_candidate_mask.tif') as mask:
        for w in windows(ref,cfg['processing']['window_rows']):
            a=ref.read(1,window=w,masked=True);v=score.read(1,window=w,masked=True)
            expected=polygon_mask(layers['boundary'],w,ref)&valid(a)&~polygon_mask(layers['water'],w,ref)&~polygon_mask(layers['historical_buffer'],w,ref)&valid(v)
            if not np.array_equal(mask.read(1,window=w)==1,expected):raise ValueError('Candidate mask readback differs from reconstructed mask')
            subset=points[(points.row>=w.row_off)&(points.row<w.row_off+w.height)]
            rr=subset.row.to_numpy()-int(w.row_off);cc=subset.col.to_numpy()
            if len(subset):
                if not expected[rr,cc].all():raise ValueError('Selected point outside candidate pool')
                np.testing.assert_array_equal(subset.hardness_score.to_numpy(),v.data[rr,cc].astype('float64'))
                x,y=rasterio.transform.xy(ref.transform,subset.row.to_numpy(),cc,offset='center')
                np.testing.assert_allclose(subset.x,x,rtol=0,atol=1e-8);np.testing.assert_allclose(subset.y,y,rtol=0,atol=1e-8)
                checked+=len(subset)
    conditions['all_in_candidate_pool']=checked==len(points)
    report['selection_qc']={k:bool(v) for k,v in conditions.items()}
    report['candidate_mask_readback']='PASS'
    if not all(conditions.values()):raise ValueError(f'Selection QC failed: {[k for k,v in conditions.items() if not v]}')


def save_samples(points,cfg,run,report):
    attributes=pd.DataFrame(points.drop(columns='geometry'))
    points.to_file(run/'samples/ns3_negative_samples.gpkg',driver='GPKG',index=False)
    csv(attributes,run/'samples/ns3_negative_samples.csv')
    for key in ('random','hard'):
        points[points.negative_source==key].to_file(run/f'samples/ns3_{key}_negatives.gpkg',driver='GPKG',index=False)
    back=gpd.read_file(run/'samples/ns3_negative_samples.gpkg')
    text=pd.read_csv(run/'samples/ns3_negative_samples.csv')
    pd.testing.assert_frame_equal(pd.DataFrame(back.drop(columns='geometry')),text,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-10)
    pd.testing.assert_frame_equal(attributes,text,check_dtype=False,check_exact=False,rtol=1e-12,atol=1e-10)
    if back.crs!=points.crs or not back.geometry.equals(points.geometry):raise ValueError('GPKG geometry/CRS readback differs')
    for key in ('random','hard'):
        subset=gpd.read_file(run/f'samples/ns3_{key}_negatives.gpkg')
        if list(subset.negative_id)!=list(points.loc[points.negative_source==key,'negative_id']):raise ValueError('Subset readback differs')
    report['gpkg_csv_consistency']='PASS'
    rows=[]
    for key in ('random','hard','all'):
        selected=points if key=='all' else points[points.negative_source==key]
        # all uses the eligible sampling union; the extreme pool was never eligible.
        pool=report['pool_counts']['random']+report['pool_counts']['hard'] if key=='all' else report['pool_counts'][key]
        v=selected.hardness_score
        rows.append(dict(group=key,pool_count=pool,sample_count=len(v),sampling_fraction=len(v)/pool if pool else np.nan,
            score_min=v.min(),score_max=v.max(),score_mean=v.mean(),score_median=v.median(),score_q25=v.quantile(.25),score_q75=v.quantile(.75)))
    csv(pd.DataFrame(rows),run/'statistics/sampling_summary.csv')
