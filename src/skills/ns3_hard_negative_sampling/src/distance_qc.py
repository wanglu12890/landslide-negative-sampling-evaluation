import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.strtree import STRtree
from .utils import csv


def distance_statistics(points,layers,cfg,run,report):
    unit=report['grid']['unit_to_m']
    for name,geometries in [('landslide',layers['inventory'].loc[layers['inventory'].ever_landslide==1,'geometry']),('buffer',layers['historical_buffer'].geometry)]:
        tree=STRtree(geometries.to_numpy())
        indices,distances=tree.query_nearest(points.geometry.to_numpy(),return_distance=True,all_matches=False)
        values=np.empty(len(points));values[indices[0]]=distances*unit
        if not np.isfinite(values).all():raise ValueError('Invalid nearest historical geometry distance')
        points[f'distance_to_{name}_m']=values
    nearest=[];historical=[]
    for key in ('random','hard','all'):
        part=points if key=='all' else points[points.negative_source==key]
        for kind in ('landslide','buffer'):
            v=part[f'distance_to_{kind}_m']
            historical.append(dict(group=key,distance_type=kind,count=len(v),min=v.min(),q05=v.quantile(.05),q25=v.quantile(.25),median=v.median(),mean=v.mean(),max=v.max()))
        if not cfg['qc']['check_nearest_neighbor']:continue
        n=len(part)
        distances=cKDTree(part[['x','y']].to_numpy()*unit).query(part[['x','y']].to_numpy()*unit,k=2)[0][:,1] if n>1 else np.array([])
        row=dict(group=key,sample_count=n,defined=n>1,min=float(distances.min()) if n>1 else np.nan,
            q05=float(np.quantile(distances,.05)) if n>1 else np.nan,q25=float(np.quantile(distances,.25)) if n>1 else np.nan,
            median=float(np.median(distances)) if n>1 else np.nan,mean=float(distances.mean()) if n>1 else np.nan)
        for t in cfg['qc']['neighbor_thresholds_m']:
            suffix=format(t,'.10g').replace('.','_')
            row[f'count_lt_{suffix}m']=int((distances<t).sum()) if n>1 else 0
            row[f'ratio_lt_{suffix}m']=float((distances<t).mean()) if n>1 else np.nan
        nearest.append(row)
        t=cfg['qc']['cluster_warning_distance_m']
        if n>1 and (distances<t).mean()>cfg['qc']['cluster_warning_fraction']:
            report['warnings'].append(f'{key}: {(distances<t).mean():.2%} nearest neighbors < {t} m exceed configured descriptive warning fraction; inspect spatial clustering, no automatic resampling.')
    csv(pd.DataFrame(nearest,columns=list(nearest[0]) if nearest else ['group','sample_count','defined']),run/'statistics/nearest_neighbor_qc.csv')
    csv(pd.DataFrame(historical),run/'statistics/historical_distance_summary.csv')
    return points
