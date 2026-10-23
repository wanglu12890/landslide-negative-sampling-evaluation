"""Ordinary paired event percentile bootstrap, fixed unified prediction fields."""
from pathlib import Path
import json,hashlib,platform,time
import numpy as np,pandas as pd
import argparse
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-root',type=Path,default=ROOT/'data/evaluation')
parser.add_argument('--output-root',type=Path,default=ROOT/'outputs/event_bootstrap')
args=parser.parse_args();U=args.data_root.resolve();O=args.output_root.resolve();O.mkdir(parents=True,exist_ok=True)
B=10000; BUDGETS=np.array([.05,.10,.20,.30]); MASTER=20261023
results=pd.read_csv(ROOT/'results/seed_results.csv'); previous=pd.read_csv(ROOT/'results/vs_ns3.csv'); rows=[]; hashes={}; started=time.time()
for seed in [42,43,44]:
 arrays={}
 for pool in ['NS3','H100','Q60']:
  path=U/f'{pool}_seed{seed}_event_ranks.npz'; hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
  with np.load(path) as z: arrays[pool]={k:z[k].copy() for k in z.files}
 ref=arrays['NS3']; n=len(ref['event_rank']); assert n==22503
 nr=results[(results.pool=='NS3')&(results.seed==seed)].iloc[0]; N=int(nr.n_valid)
 for pool in ['H100','Q60']:
  for key in ['event_index','event_row','event_col']: assert np.array_equal(arrays[pool][key],ref[key]), (seed,pool,key)
  rr=results[(results.pool==pool)&(results.seed==seed)].iloc[0]; assert int(rr.n_valid)==N and int(rr.n_events)==n
 thresholds=np.floor(BUDGETS*N).astype(int)
 baseline=ref['event_rank'][:,None]<thresholds[None,:]
 delta=np.concatenate([(arrays[pool]['event_rank'][:,None]<thresholds).astype(np.int8)-baseline.astype(np.int8) for pool in ['H100','Q60']],axis=1)
 # Identical bootstrap event indices across comparisons and all budgets within a seed.
 rngseed=MASTER+seed; rng=np.random.default_rng(rngseed); draws=np.empty((B,8))
 for start in range(0,B,100):
  stop=min(start+100,B); ids=rng.integers(0,n,size=(stop-start,n)); draws[start:stop]=delta[ids].mean(axis=1)*100
 lo,hi=np.quantile(draws,[.025,.975],axis=0,method='linear'); point=delta.mean(axis=0)*100
 assert np.all(lo<=hi)
 # Paired estimator identity and antisymmetry checks.
 assert np.allclose(point,100*delta.sum(axis=0)/n)
 assert np.allclose(np.quantile(-draws,[.025,.975],axis=0),np.stack([-hi,-lo]))
 np.savez_compressed(O/f'seed{seed}_draws.npz',delta_capture_pp=draws,columns=np.array([f'{p}_minus_NS3_top{int(b*100)}' for p in ['H100','Q60'] for b in BUDGETS]))
 pd.DataFrame({'event_index':ref['event_index'],'row':ref['event_row'],'col':ref['event_col'],**{f'{p}_minus_NS3_top{int(b*100)}':delta[:,i*4+j] for i,p in enumerate(['H100','Q60']) for j,b in enumerate(BUDGETS)}}).to_csv(O/f'seed{seed}_event_paired_differences.csv',index=False)
 for i,pool in enumerate(['H100','Q60']):
  for j,b in enumerate(BUDGETS):
   k=i*4+j; orig=previous[(previous.pool==pool)&(previous.seed==seed)].iloc[0][f'capture_top{int(b*100)}']*100
   assert np.isclose(point[k],orig,atol=1e-10)
   rows.append(dict(pool=pool,reference='NS3',training_seed=seed,budget=b,selected_pixels=int(thresholds[j]),n_valid=N,n_events=n,delta_capture_pp=point[k],ci_lower_pp=lo[k],ci_upper_pp=hi[k],excludes_zero=bool(lo[k]>0 or hi[k]<0),iterations=B,bootstrap_seed=rngseed,method='paired event percentile bootstrap',quantile_method='linear'))
 print('seed',seed,'complete',flush=True)
d=pd.DataFrame(rows);d.to_csv(O/'paired_event_bootstrap.csv',index=False)
meta=dict(training_seeds=[42,43,44],budgets=BUDGETS.tolist(),iterations=B,master_seed=MASTER,rng='numpy.default_rng PCG64',seed_rule='master_seed + training_seed',interval='marginal 95% percentile; no multiplicity adjustment',resampling='n events with replacement; same indices across budgets and comparisons within seed',numpy=np.__version__,pandas=pd.__version__,python=platform.python_version(),input_sha256=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_seconds=time.time()-started,tests=['aligned event ID and pixel coordinates','same N and n across pairs','point estimates match unified table','paired estimator identity','swap-comparison antisymmetry'])
(O/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
lines=['# 各seed、四预算的成对事件Bootstrap','', '已完成24组比较（2策略×3seed×4预算），每组10000次成对事件重采样。使用统一划分后的固定预测场与同一22503事件，区间为边际95%百分位区间，未做多重比较校正。差值方向为策略−NS3，单位百分点。','', '|比较|训练seed|Top-5% Δ [95% CI]|Top-10% Δ [95% CI]|Top-20% Δ [95% CI]|Top-30% Δ [95% CI]|','|---|---:|---|---|---|---|']
for pool in ['H100','Q60']:
 for seed in [42,43,44]:
  sub=d[(d.pool==pool)&(d.training_seed==seed)].sort_values('budget');cells=[f'{r.delta_capture_pp:+.3f} [{r.ci_lower_pp:+.3f}, {r.ci_upper_pp:+.3f}]' for r in sub.itertuples()]
  lines.append('|'+f'{pool}−NS3|{seed}|'+'|'.join(cells)+'|')
lines+=['','## 方法与验收','', '每次有放回抽取22503个事件索引，两个模型使用同一批索引；同seed四预算及两比较也共用索引。固定预测排名和覆盖区，不在bootstrap内重新训练或重新选阈值。严格选floor(bN)像元，得分并列按像元索引处理。各seed的bootstrap RNG seed分别为20261065、20261066、20261067；按保存事件顺序进行10000次抽样。不同训练seed不合并事件，不把同一台风重复计为独立事件。','', '已检查事件ID及行列坐标对齐、有效域及分母一致、点估计精确复现统一结果表、成对差值恒等式及交换比较方向后的区间对称性。保存24行CSV、各seed逐事件差值、全部bootstrap差值draws、输入SHA256及版本元数据。没有保存庞大的重采样索引；固定输入顺序、脚本及RNG seed可复现抽样。','', '## 解释边界与下一步','', '这些区间仅度量固定预测场下的事件抽样不确定性，采用事件可独立重采样的假设；同一场台风中的空间聚集可能使区间偏窄。区间不是训练seed变异，不是跨灾害泛化证据；区间包含0也不等于两个策略等效。24个区间未作同时覆盖校正，应保持探索性解释。','', '1. 下一步优先做H100−NS3、Q60−NS3的空间块Bootstrap，保留相同seed和四预算；预先固定网格原点及尺度，记录各块事件数和event→block映射，按抽中块携带全部事件并以重采样事件总数作分母。','2. 检查多个合理尺度（如1/2/5公里）及有效block数量，报告尺度敏感性，不能按区间显著与否选尺度。空间块区间与本事件级区间并排展示。','3. 随后更新论文Table 3为真实差值和区间端点，训练seed均值/标准差另表报告；仅按统计证据保留Q60的预算及seed条件性优势，H100内部预警能力也按seed区别描述。','', '结果路径：experiments/unified_split/paired_event_bootstrap/paired_event_bootstrap.csv、seed*_draws.npz、seed*_event_paired_differences.csv、metadata.json。原始输入未改动。']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(d.to_string(index=False))
