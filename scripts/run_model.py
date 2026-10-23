"""Run the original RF pipeline with canonical split and published ID guards."""
from pathlib import Path
import argparse,sys,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-key',required=True);p.add_argument('--config',type=Path);args=p.parse_args()
records=json.loads((ROOT/'manifests/runs.json').read_text());record=next(r for r in records if r['run_key']==args.run_key)
sys.path.insert(0,str(ROOT/'src/skills/lsm_modeling'));sys.path.insert(0,str(ROOT/'src'))
import run_lsm as runner
from src.dataset_builder import split_dataset as canonical_split
expected=pd.read_csv(ROOT/'manifests/splits'/args.run_key/'train_ids.csv');held=pd.read_csv(ROOT/'manifests/splits'/args.run_key/'test_ids.csv')
def checked_split(data,cfg,run,report):
 train,test=canonical_split(data,cfg,run,report)
 assert list(train.sample_id)==list(expected.sample_id), 'Training IDs/order differ from recorded experiment'
 assert list(test.sample_id)==list(held.sample_id), 'Test IDs/order differ from recorded experiment'
 assert not set(train.sample_id)&set(test.sample_id)
 assert report['split']['shared_train_test_cell_count']==0
 report['split']['release_id_guard']='canonical split IDs and order match saved experiment'
 return train,test
runner.split_dataset=checked_split
print(runner.run((args.config or ROOT/record['model_config']).resolve()))
