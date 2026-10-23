"""Check the isolated release without private data or training."""
from pathlib import Path
import argparse,ast,json,hashlib,importlib,sys,subprocess,tempfile
import numpy as np,pandas as pd,yaml
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--self-test',action='store_true');args=p.parse_args();checks={}
for path in ROOT.rglob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
checks['python_syntax']='PASS';sys.path.insert(0,str(ROOT/'src'))
for package in ['ns1_random_sampling','ns2_low_susceptibility_sampling','ns3_hard_negative_sampling','lsm_modeling']:
 for path in (ROOT/'src/skills'/package/'src').glob('*.py'):
  if path.name!='__init__.py':
   module=importlib.import_module('skills.'+package+'.src.'+path.stem);assert Path(module.__file__).resolve().is_relative_to(ROOT)
checks['source_module_imports']='PASS'
for r in json.loads((ROOT/'manifests/runs.json').read_text()):
 for field in ['sampling_config','model_config']:
  cfg=yaml.safe_load((ROOT/r[field]).read_text());assert (ROOT/r[field]).parent.joinpath(cfg['project_root']).resolve()==ROOT
 folder=ROOT/'manifests/splits'/r['run_key'];tr=pd.read_csv(folder/'train_ids.csv');te=pd.read_csv(folder/'test_ids.csv');pool=pd.read_csv(folder/'pool_ids.csv')
 assert pool.sample_id.is_unique and not set(tr.sample_id)&set(te.sample_id)
 assert set(tr.sample_id)|set(te.sample_id)==set(pool.sample_id)
 assert len(pool)==1600 and len(tr)==1120 and len(te)==480
 assert (tr.label.value_counts()==560).all() and (te.label.value_counts()==240).all()
 assert set(pool.columns)=={'record_order','sample_id','label','sample_type','origin'}
 from sklearn.model_selection import train_test_split
 a,b=train_test_split(pool,test_size=.3,random_state=r['seed'],stratify=pool.label)
 assert list(a.sample_id)==list(tr.sample_id) and list(b.sample_id)==list(te.sample_id)
 fit=pd.read_csv(folder/'scorer_fit_positive_ids.csv');assert len(fit)==560 and fit.sample_id.is_unique
 assert set(fit.sample_id)==set(tr.loc[tr.label==1,'sample_id']) and not set(fit.sample_id)&set(te.loc[te.label==1,'sample_id'])
 cfg=yaml.safe_load((ROOT/r['model_config']).read_text());assert cfg['model']['params']['n_estimators']==500 and cfg['split']['stratify'] is True and cfg['split']['random_seed']==r['seed']
checks['all_18_run_ids_and_configs']='PASS'
for r in pd.read_csv(ROOT/'manifests/file_provenance.csv').to_dict('records'):
 assert hashlib.sha256((ROOT/r['destination']).read_bytes()).hexdigest()==r['copied_sha256'],r['destination']
checks['source_and_result_file_hashes']='PASS'
s=pd.read_csv(ROOT/'results/seed_results.csv');main=pd.read_csv(ROOT/'results/seed42_metrics.csv')
for r in main[main.pool.isin(s.pool)].itertuples():
 q=s[(s.pool==r.pool)&(s.seed==42)].iloc[0]
 for col in ['internal_auc','sr_auc','capture_top5','capture_top10','capture_top20','capture_top30']:assert np.isclose(getattr(r,col),q[col],atol=1e-12)
e=pd.read_csv(ROOT/'results/event_bootstrap.csv');b=pd.read_csv(ROOT/'results/spatial_bootstrap.csv');assert len(e)==24 and len(b)==72
assert (e.iterations==10000).all() and (b.iterations==10000).all();checks['published_result_integrity']='PASS'
for name in ['run_sampling.py','run_model.py','evaluate_rasters.py','event_bootstrap.py','spatial_bootstrap.py']:
 proc=subprocess.run([sys.executable,str(ROOT/'scripts'/name),'--help'],cwd=ROOT,capture_output=True,text=True);assert proc.returncode==0,(name,proc.stderr)
checks['all_cli_help']='PASS'
if args.self_test:
 import rasterio
 from rasterio.transform import from_origin
 parent=ROOT/'outputs';parent.mkdir(exist_ok=True)
 with tempfile.TemporaryDirectory(dir=parent) as tmp:
  t=Path(tmp);profile=dict(driver='GTiff',height=10,width=10,count=1,dtype='float32',crs='EPSG:32649',transform=from_origin(0,100,10,10))
  scores=np.arange(100,dtype='float32')[::-1].reshape(10,10)/100;scores.flat[:10]=1
  for name,array in [('domain.tif',np.ones((10,10),dtype='float32')),('score.tif',scores)]:
   with rasterio.open(t/name,'w',**profile) as dst:dst.write(array,1)
  pd.DataFrame({'row':[0,0,9],'col':[1,5,5],'common_valid':[True]*3}).to_csv(t/'events.csv',index=False)
  cfg={'domain':'domain.tif','events':'events.csv','output':'out','predictions':[{'pool':'toy','seed':42,'raster':'score.tif'}]};(t/'config.json').write_text(json.dumps(cfg))
  proc=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_rasters.py'),'--config',str(t/'config.json')],cwd=ROOT,capture_output=True,text=True);assert proc.returncode==0,proc.stderr
  result=pd.read_csv(t/'out/ranking_results.csv').iloc[0]
  assert np.isclose(result.capture_top5,1/3) and np.isclose(result.capture_top10,2/3)
  assert np.isclose(result.sr_auc,np.mean((100-np.array([1,5,95])-1)/100))
  assert np.isclose(result.lift_top5,(1/3)/.05)
 checks['synthetic_ties_exact_budget_lift_complete_auc']='PASS'
 delta=np.array([[1,-1],[0,1],[-1,1],[1,0]]);group=np.array([0,0,1,2]);chosen=[0,2,0];counts=np.bincount(group);sums=np.zeros((3,2),dtype=int);np.add.at(sums,group,delta)
 expanded=np.concatenate([np.flatnonzero(group==k) for k in chosen]);assert np.allclose(delta[expanded].mean(0),sums[chosen].sum(0)/counts[chosen].sum());checks['paired_cluster_variable_denominator']='PASS'
(ROOT/'outputs').mkdir(exist_ok=True);(ROOT/'outputs/verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
