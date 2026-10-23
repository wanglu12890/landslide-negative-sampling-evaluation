"""Exact-area evaluation using the paper's lexsort and complete step integral."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd,rasterio
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);args=p.parse_args()
cfg=json.loads(args.config.read_text());base=args.config.resolve().parent
resolve=lambda x:(base/x).resolve()
with rasterio.open(resolve(cfg['domain'])) as r:
 domain=r.read(1)==1;width=r.width;transform=r.transform;crs=r.crs
flat=np.flatnonzero(domain.ravel());n=len(flat)
e=pd.read_csv(resolve(cfg['events']));e=e[e.common_valid==True];ef=(e.row.to_numpy()*width+e.col.to_numpy()).astype(int);pos=np.searchsorted(flat,ef);assert np.array_equal(flat[pos],ef)
out=resolve(cfg['output']);out.mkdir(parents=True,exist_ok=True);rows=[]
for item in cfg['predictions']:
 pool,seed=item['pool'],item['seed']
 with rasterio.open(resolve(item['raster'])) as r:
  assert r.width==width and r.transform==transform and r.crs==crs
  a=r.read(1,masked=True);assert not np.ma.getmaskarray(a).ravel()[flat].any();scores=a.data.ravel()[flat]
 assert np.isfinite(scores).all()
 order=np.lexsort((flat,-scores));rank=np.empty(n,dtype=np.int32);rank[order]=np.arange(n);er=rank[pos]
 ev=scores[pos];med=float(np.median(ev));delta=float((scores>=med).mean())
 row=dict(pool=pool,seed=seed,n_valid=n,n_events=len(e),sr_auc=float(np.mean((n-er-1)/n)),delta=delta)
 for b in [.05,.1,.2,.3]:
  k=int(b*n);cr=float((er<k).mean());row[f'capture_top{int(b*100)}']=cr;row[f'lift_top{int(b*100)}']=cr/(k/n)
 rows.append(row)
 np.savez_compressed(out/f'{pool}_seed{seed}_event_ranks.npz',event_rank=er,event_row=e.row.to_numpy(),event_col=e.col.to_numpy(),event_index=e.index.to_numpy())
pd.DataFrame(rows).to_csv(out/'ranking_results.csv',index=False)
