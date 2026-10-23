import geopandas as gpd
import pandas as pd
import numpy as np
from .utils import csv


def inspect_inventory(path, cfg, run, report):
    c = cfg['inventory']; fields = c['temporal_fields']
    # Read only the formal temporal window, not later partially annotated fields.
    columns = fields + ([c['id_field']] if c.get('id_field') else [])
    raw = gpd.read_file(path, columns=columns, fid_as_index=True)
    report['inventory_polygon_count'] = len(raw)
    missing_fields = [f for f in fields if f not in raw]
    report['missing_temporal_fields'] = missing_fields
    if missing_fields:
        raise ValueError(f'Missing formal temporal fields: {missing_fields}')
    if not len(raw):
        raise ValueError('Inventory is empty')
    out = raw.copy()
    out.insert(0, 'feature_id', raw.index.to_numpy())
    warnings, invalid, sequences = [], [], []
    allowed = set(c['allowed_states']); historical = set(c['landslide_states'])
    new = set(c['new_states']); uncertain = set(c['uncertain_states'])
    reactivation = set(c['reactivation_states']); expansion = set(c['expansion_states'])
    for f in fields:
        out[f] = raw[f].map(lambda v: None if pd.isna(v) else str(v).strip().upper())
    for fid, row in out.iterrows():
        seen, seen_new, first = False, 0, None
        values = [row[f] for f in fields]
        for f,v in zip(fields, values):
            def warn(kind):
                warnings.append(dict(feature_id=int(fid), year_field=f, state=v, warning_type=kind))
            if v is None or v == '' or v not in allowed:
                kind = 'missing_state' if v is None or v == '' else 'illegal_state'
                invalid.append(dict(feature_id=int(fid), object_id=row.get(c.get('id_field'), None), year_field=f,
                                    raw_value=raw.loc[fid,f], normalized_value=v, issue=kind))
                warn(kind)
                continue
            if v in uncertain:
                warn('unknown_observation')
            if v in new:
                seen_new += 1
                if first is None:
                    first = f
                if seen_new > 1:
                    warn('multiple_new_states')
            if v in c['non_detection_states'] and seen_new:
                warn('non_detection_after_new')
            if v in historical and v not in new and not seen:
                warn('historical_state_without_prior_in_window')
                if v in reactivation:
                    warn('reactivation_without_prior_in_window')
                if v in expansion:
                    warn('expansion_without_prior_in_window')
                if v in c['combined_states']:
                    warn('combined_state_without_prior_in_window')
            seen |= v in historical
        sequences.append(dict(feature_id=int(fid), **{f:row[f] for f in fields}, first_N_year=first,
            ever_landslide=int(any(v in historical for v in values)),
            reactivation_flag=int(any(v in reactivation for v in values)), expansion_flag=int(any(v in expansion for v in values)),
            unknown_count=sum(v in uncertain for v in values)))
    seq = pd.DataFrame(sequences)
    issues = pd.DataFrame(invalid, columns=['feature_id','object_id','year_field','raw_value','normalized_value','issue'])
    warning = pd.DataFrame(warnings, columns=['feature_id','year_field','state','warning_type'])
    counts = []
    for f in fields:
        observed = out[f].fillna('<MISSING>').replace('', '<MISSING>').value_counts()
        for state in list(c['allowed_states']) + sorted(set(observed.index)-allowed):
            counts.append(dict(year_field=f, state=state, count=int(observed.get(state,0)), ratio=float(observed.get(state,0)/len(out))))
    yearly = pd.DataFrame(counts)
    transitions = []
    for a,b in zip(fields[:-1], fields[1:]):
        for (sa,sb),n in out[[a,b]].fillna('<MISSING>').groupby([a,b], dropna=False).size().items():
            transitions.append(dict(from_field=a,to_field=b,from_state=sa,to_state=sb,count=int(n)))
    # Additional effective-observation transitions bridge U but never bridge illegal/missing states.
    effective = []
    for fid, row in out.iterrows():
        previous = None
        for f in fields:
            v = row[f]
            if v not in allowed:
                previous = None
            elif v not in uncertain:
                if previous:
                    effective.append(dict(from_field=previous[0],to_field=f,from_state=previous[1],to_state=v))
                previous = (f,v)
    eff = pd.DataFrame(effective, columns=['from_field','to_field','from_state','to_state'])
    eff = eff.groupby(list(eff.columns), dropna=False).size().reset_index(name='count')
    for name, frame in [('inventory_temporal_sequences',seq),('inventory_sequence_warnings',warning),
                        ('yearly_state_counts',yearly),('state_transition_counts',pd.DataFrame(transitions)),
                        ('effective_observation_transition_counts',eff),('inventory_invalid_states',issues)]:
        csv(frame, run/f'inventory/{name}.csv')
    for col in ('ever_landslide','reactivation_flag','expansion_flag','unknown_count'):
        out[col] = seq[col].to_numpy()
    out[c['derived_ever_field']] = seq.ever_landslide.to_numpy()
    out['state_qc_valid'] = ~out.feature_id.isin(issues.feature_id)
    out.to_file(run/'inventory/inventory_qc.gpkg', driver='GPKG', index=False)
    report.update(invalid_state_count=int((issues.issue=='illegal_state').sum()), missing_state_count=int((issues.issue=='missing_state').sum()),
        combined_state_occurrences=int(out[fields].isin(c['combined_states']).sum().sum()),
        combined_state_polygon_count=int(out[fields].isin(c['combined_states']).any(axis=1).sum()),
        ever_landslide_count=int(seq.ever_landslide.sum()), ever_landslide_field=c['derived_ever_field'],
        sequence_warning_count=len(warning), warning_polygon_count=int(warning.feature_id.nunique()),
        warning_types={str(k):int(v) for k,v in warning.warning_type.value_counts().items()},
        inventory_states_valid=issues.empty,
        inventory_note='Derived ever_landslide is provisional when illegal/missing states exist; no candidate analysis allowed.')
    return out
