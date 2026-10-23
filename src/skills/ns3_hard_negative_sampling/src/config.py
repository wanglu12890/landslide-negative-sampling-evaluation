from pathlib import Path
import math
import yaml


def load_config(path):
    cfg=yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    root=Path(cfg['project_root'])
    if not root.is_absolute():root=path.parent/root
    root=root.resolve()
    keys={'positive','inventory','historical_buffer','boundary','water','reference_raster','hardness_raster'}
    if set(cfg['inputs'])!=keys:raise ValueError('Exactly seven required inputs must be configured')
    paths={k:(root/v['path']).resolve() for k,v in cfg['inputs'].items()}
    comparisons={k:(root/v['path']).resolve() for k,v in cfg.get('comparison_samples',{}).items()}
    all_inputs={**paths,**{'comparison:'+k:v for k,v in comparisons.items()}}
    for p in all_inputs.values():
        if not p.is_file():raise FileNotFoundError(p)
    inv=cfg['inventory']; fs=inv['temporal_fields']; allowed=set(inv['allowed_states'])
    if not fs or len(fs)!=len(set(fs)):raise ValueError('Temporal fields must be unique, nonempty, chronological')
    if any(not isinstance(s,str) or s!=s.strip().upper() for s in allowed):raise ValueError('Normalize configured state sets')
    sets=[set(inv[k]) for k in ('landslide_states','non_detection_states','uncertain_states')]
    if set.union(*sets)!=allowed or sum(map(len,sets))!=len(allowed):raise ValueError('Semantic sets must partition allowed states')
    for k in ('new_states','reactivation_states','expansion_states','combined_states'):
        if not set(inv[k])<=sets[0]:raise ValueError(f'{k} must belong to landslide_states')
    h=cfg['hard_negative']; lo,hi=h['lower_quantile'],h['upper_quantile']
    if not 0<lo<hi<1:raise ValueError('Require 0 < lower_quantile < upper_quantile < 1')
    s=cfg['sampling']
    if s['replace'] is not False:raise ValueError('Sampling must be without replacement')
    if not all(math.isfinite(s[k]) and 0<=s[k]<=1 for k in ('random_ratio','hard_ratio')) or not math.isclose(s['random_ratio']+s['hard_ratio'],1,abs_tol=1e-12):
        raise ValueError('Random and hard ratios must sum to one')
    if not isinstance(s['seed'],int) or s['seed']<0:raise ValueError('Seed must be a nonnegative integer')
    total=s['total_negative_count']
    if total!='auto' and (not isinstance(total,int) or total<=0):raise ValueError('total_negative_count must be auto or positive integer')
    ex=cfg['exclusion']; reasons={'outside_boundary','reference_nodata','water','historical_landslide_buffer','hardness_nodata'}
    if set(ex['priority'])!=reasons or len(ex['priority'])!=len(reasons) or set(ex['codes'])!=reasons:raise ValueError('Invalid exclusion priority/codes')
    if len(set(ex['codes'].values()))!=len(reasons) or any(not isinstance(v,int) or not 1<=v<255 for v in ex['codes'].values()):raise ValueError('Exclusion codes must be unique in 1..254')
    b=cfg['buffer_qc']
    if not 0<=b['tolerance_m']<b['expected_distance_m'] or b['missing_area_tolerance_m2']<0:raise ValueError('Invalid buffer tolerance')
    if cfg['processing']['window_rows']<=0 or cfg['plots']['dpi']<=0:raise ValueError('Processing sizes must be positive')
    qc=cfg['qc']
    if qc['quantile_warning_tolerance']<0 or any(d<=0 for d in qc['neighbor_thresholds_m']):raise ValueError('Invalid QC thresholds')
    if not 0<=qc['cluster_warning_fraction']<=1 or qc['cluster_warning_distance_m']<=0:raise ValueError('Invalid clustering warning settings')
    out=(root/cfg['output']['root']).resolve()
    if any(out==p.parent or p.parent in out.parents for p in all_inputs.values()):raise ValueError('Output must be outside input directories')
    return cfg,paths,comparisons,all_inputs,out
