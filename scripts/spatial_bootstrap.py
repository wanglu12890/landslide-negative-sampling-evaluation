"""Paired occupied-spatial-block bootstrap at predeclared 1/2/5 km scales."""
from pathlib import Path
import sys,json,hashlib,platform,time
import numpy as np,pandas as pd,rasterio
import argparse
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--events',type=Path,required=True)
parser.add_argument('--domain',type=Path,required=True)
parser.add_argument('--event-bootstrap-root',type=Path,required=True)
parser.add_argument('--output-root',type=Path,default=ROOT/'outputs/spatial_bootstrap')
args=parser.parse_args();DOMAIN=args.domain;EVENTS=args.events;O=args.output_root.resolve();O.mkdir(parents=True,exist_ok=True)
B=10000;SCALES=[1000,2000,5000];MASTER=202610230;started=time.time();hashes={}
with rasterio.open(DOMAIN) as r:
 assert r.crs.is_projected and r.crs.linear_units=='metre';crs=r.crs.to_string();trans=r.transform
 origin=(np.floor(r.bounds.left/5000)*5000,np.floor(r.bounds.bottom/5000)*5000)
e=pd.read_csv(EVENTS);e=e[e.common_valid==True].copy();n=len(e);assert n==22503
rawcols=np.floor((e.x.to_numpy()-trans.c)/trans.a).astype(int); rawrows=np.floor((e.y.to_numpy()-trans.f)/trans.e).astype(int)
assert np.array_equal(rawcols,e.col) and np.array_equal(rawrows,e.row)
event=pd.read_csv(args.event_bootstrap_root/'paired_event_bootstrap.csv'); rows=[];stats=[]
for scale in SCALES:
 bx=np.floor((e.x.to_numpy()-origin[0])/scale).astype(int);by=np.floor((e.y.to_numpy()-origin[1])/scale).astype(int)
 pairs=np.column_stack([bx,by]);unique,group=np.unique(pairs,axis=0,return_inverse=True);K=len(unique);counts=np.bincount(group,minlength=K); assert counts.sum()==n and counts.min()>0
 mapping=pd.DataFrame({'event_id':e.event_id.to_numpy(),'event_index':e.index.to_numpy(),'x':e.x.to_numpy(),'y':e.y.to_numpy(),'row':e.row.to_numpy(),'col':e.col.to_numpy(),'block_id':group,'block_x':bx,'block_y':by})
 mapping.to_csv(O/f'event_block_mapping_{scale}m.csv',index=False)
 blocktable=pd.DataFrame({'block_id':np.arange(K),'block_x':unique[:,0],'block_y':unique[:,1],'event_count':counts,'xmin':origin[0]+unique[:,0]*scale,'ymin':origin[1]+unique[:,1]*scale});blocktable['xmax']=blocktable.xmin+scale;blocktable['ymax']=blocktable.ymin+scale;blocktable.to_csv(O/f'blocks_{scale}m.csv',index=False)
 stats.append(dict(scale_m=scale,occupied_blocks=K,n_events=n,min_events=int(counts.min()),median_events=float(np.median(counts)),max_events=int(counts.max()),singleton_blocks=int((counts==1).sum()),event_share_largest=float(counts.max()/n)))
 for seed in [42,43,44]:
  d=pd.read_csv(args.event_bootstrap_root/f'seed{seed}_event_paired_differences.csv');assert np.array_equal(d.event_index,e.index) and np.array_equal(d.row,e.row) and np.array_equal(d.col,e.col)
  cols=[f'{p}_minus_NS3_top{b}' for p in ['H100','Q60'] for b in [5,10,20,30]];delta=d[cols].to_numpy(); sums=np.zeros((K,8),dtype=np.int64);np.add.at(sums,group,delta)
  assert np.array_equal(sums.sum(0),delta.sum(0))
  rngseed=MASTER+scale*10+seed;rng=np.random.default_rng(rngseed);draws=np.empty((B,8));denoms=np.empty(B,dtype=np.int64)
  for start in range(0,B,100):
   stop=min(B,start+100);idx=rng.integers(0,K,(stop-start,K));den=counts[idx].sum(axis=1);assert (den>0).all();draws[start:stop]=sums[idx].sum(axis=1)/den[:,None]*100;denoms[start:stop]=den
  # Explicit event expansion check: repeating a block repeats every one of its events.
  selected=np.array([0,K-1,0]);expanded=np.concatenate([np.flatnonzero(group==g) for g in selected]);assert np.allclose(delta[expanded].mean(0),sums[selected].sum(0)/counts[selected].sum())
  lo,hi=np.quantile(draws,[.025,.975],axis=0,method='linear');point=delta.mean(0)*100
  assert np.allclose(np.quantile(-draws,[.025,.975],axis=0),np.stack([-hi,-lo]))
  np.savez_compressed(O/f'draws_{scale}m_seed{seed}.npz',delta_capture_pp=draws,resampled_event_count=denoms,columns=np.array(cols))
  for i,pool in enumerate(['H100','Q60']):
   for j,b in enumerate([.05,.1,.2,.3]):
    k=i*4+j;er=event[(event.pool==pool)&(event.training_seed==seed)&np.isclose(event.budget,b)].iloc[0];assert np.isclose(point[k],er.delta_capture_pp)
    ew=er.ci_upper_pp-er.ci_lower_pp;sw=hi[k]-lo[k]
    rows.append(dict(pool=pool,reference='NS3',training_seed=seed,budget=b,scale_m=scale,occupied_blocks=K,n_events=n,delta_capture_pp=point[k],event_ci_lower_pp=er.ci_lower_pp,event_ci_upper_pp=er.ci_upper_pp,spatial_ci_lower_pp=lo[k],spatial_ci_upper_pp=hi[k],width_ratio_spatial_event=sw/ew,spatial_excludes_zero=bool(lo[k]>0 or hi[k]<0),event_excludes_zero=bool(er.excludes_zero),iterations=B,bootstrap_seed=rngseed,min_resampled_events=int(denoms.min()),max_resampled_events=int(denoms.max())))
  print(scale,seed,'complete',flush=True)
  fp=args.event_bootstrap_root/f'seed{seed}_event_paired_differences.csv';hashes[fp.name]=hashlib.sha256(fp.read_bytes()).hexdigest()
table=pd.DataFrame(rows);table.to_csv(O/'spatial_vs_event_intervals.csv',index=False);pd.DataFrame(stats).to_csv(O/'block_diagnostics.csv',index=False)
metadata=dict(scales_m=SCALES,iterations=B,master_seed=MASTER,seed_rule='master + scale_m*10 + training_seed',rng='numpy.default_rng PCG64',crs=crs,origin_xy_m=origin,grid='floor((x-origin_x)/scale), floor((y-origin_y)/scale); lower closed upper open',sampling_population='occupied blocks only, conditional on event-bearing blocks; empty blocks not sampled',estimator='sum paired event differences in sampled blocks / sum event counts in sampled blocks; repeated blocks repeat all events',interval='marginal 95% percentile; quantile linear; no multiplicity adjustment',scale_rationale='predeclared 1/2/5 km grids examine increasing clustering scales; not estimated decorrelation lengths or validated optimal blocks',versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,rasterio=rasterio.__version__),input_sha256=hashes,domain_sha256=hashlib.sha256(DOMAIN.read_bytes()).hexdigest(),events_sha256=hashlib.sha256(EVENTS.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_seconds=time.time()-started,checks=['actual event coordinate to pixel match','event identity and row/col alignment','block counts sum to n','block sums equal event sums','explicit repeated-block event expansion','point estimates match event bootstrap','comparison reversal symmetry'])
(O/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
print(pd.DataFrame(stats).to_string(index=False));print(table.groupby(['pool','scale_m']).agg(excluding_zero=('spatial_excludes_zero','sum'),median_width_ratio=('width_ratio_spatial_event','median')).to_string())
