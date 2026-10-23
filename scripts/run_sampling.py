"""Run recorded sampling; corrected NS2 fits IV on saved training positives only."""
from pathlib import Path
import argparse,sys,json,importlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-key',required=True);p.add_argument('--config',type=Path);args=p.parse_args()
record=next(r for r in json.loads((ROOT/'manifests/runs.json').read_text()) if r['run_key']==args.run_key)
package=record['sampling_package'];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'src/skills'/package))
entry={'NS1':'run_ns1','NS2':'run_ns2'}.get(record['pool'],'run_ns3')
runner=importlib.import_module('skills.'+package+'.'+entry)
if record['pool']=='NS2':
 saved=pd.read_csv(ROOT/'manifests/splits'/args.run_key/'train_ids.csv');ids=set(saved.loc[saved.label==1,'sample_id']);assert len(ids)==560
 original=runner.calculate_iv
 def training_only(counts,observed,positives,cfg,reference,run,report):
  selected=positives[positives.sample_id.astype(str).isin(ids)].copy();assert len(selected)==560
  report['iv_fit_audit']={'positive_count':560,'heldout_overlap':0,'positive_ids':sorted(ids),'protocol':'training-only IV, recorded IDs'}
  selected.to_csv(run/'qc/iv_training_positives.csv',index=False)
  return original(counts,observed,selected,cfg,reference,run,report)
 runner.calculate_iv=training_only
result=runner.run((args.config or ROOT/record['sampling_config']).resolve())
if isinstance(result,tuple):
 directory,report=result
 if report['qc_status']!='PASS':raise RuntimeError(report.get('error','Sampling QC failed'))
 print(directory)
else:print(result)
