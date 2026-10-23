import numpy as np
import pandas as pd
from .utils import csv,qname


def define_pools(values,cfg,run,report):
    lower,upper=[cfg['hard_negative'][k] for k in ('lower_quantile','upper_quantile')]
    # Quantiles always use all candidates, never values copied from earlier runs.
    lo,hi=map(float,np.quantile(values,[lower,upper]))
    if not 0<=lo<hi<=1:raise ValueError(f'Hardness thresholds must be distinct and ordered: {lo}, {hi}')
    v=values.astype('float64')
    counts=dict(random=int((v<lo).sum()),hard=int(((v>=lo)&(v<hi)).sum()),extreme=int((v>=hi).sum()))
    if sum(counts.values())!=len(v):raise ValueError('Pool partition mismatch')
    for key in ('random','hard'):
        if counts[key]<report['requested_counts'][key]:raise ValueError(f'Insufficient {key} pool: {counts[key]}')
    report['thresholds']={'lower_quantile':lower,'upper_quantile':upper,'lower_score':lo,'upper_score':hi}
    report['pool_counts']=counts;report['pool_partition']='PASS'
    area=report['pixel_area_km2']
    row=dict(candidate_count=len(v),candidate_area_km2=len(v)*area,
        **{f'{k}_pool_count':n for k,n in counts.items()},**{f'{k}_pool_area_km2':n*area for k,n in counts.items()},
        **{qname(lower).lower()+'_score':lo,qname(upper).lower()+'_score':hi})
    csv(pd.DataFrame([row]),run/'statistics/candidate_pool_summary.csv')
    for q,score in ((lower,lo),(upper,hi)):
        expected=cfg['qc'].get('expected_candidate_quantiles',{}).get(qname(q).lower())
        if expected is not None and abs(score-expected)>cfg['qc']['quantile_warning_tolerance']:
            report['warnings'].append(f'{qname(q)} actual={score:.8f}, reference={expected}; exceeds tolerance. Candidate mask rebuilt from current inputs, inspect exclusion_summary.csv and spatial masks; old score not substituted.')
    return lo,hi
