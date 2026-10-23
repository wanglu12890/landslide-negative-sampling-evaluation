"""Construct independent mixed negative samples using existing baseline RF scores."""
import argparse
import importlib.metadata
import logging
from pathlib import Path
import sys
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from skills.ns3_hard_negative_sampling.src.config import load_config
from skills.ns3_hard_negative_sampling.src.utils import create_run,snapshot,write_json
from skills.ns3_hard_negative_sampling.src.inventory_qc import inspect_inventory
from skills.ns3_hard_negative_sampling.src.input_validation import validate_spatial,validate_positive
from skills.ns3_hard_negative_sampling.src.candidate_pool import build_candidates
from skills.ns3_hard_negative_sampling.src.hardness import define_pools
from skills.ns3_hard_negative_sampling.src.sampling import select_samples,validate_selection,save_samples
from skills.ns3_hard_negative_sampling.src.distance_qc import distance_statistics
from skills.ns3_hard_negative_sampling.src.overlap_qc import compare_overlap
from skills.ns3_hard_negative_sampling.src.plotting import plot_samples
from skills.ns3_hard_negative_sampling.src.report import write_reports


def run(path):
    cfg,paths,comparisons,inputs,root=load_config(path.resolve());folder=create_run(root,cfg['sampling']['seed'])
    report=dict(run_id=folder.name,experiment_name=cfg['experiment_name'],qc_status='FAIL',warnings=[],
        input_paths={k:str(v) for k,v in inputs.items()},input_snapshot=snapshot(inputs),
        versions={k:importlib.metadata.version(k) for k in ('numpy','pandas','geopandas','rasterio','shapely','pyproj','scipy','matplotlib','pyyaml')})
    (folder/'config/config_used.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
    try:
        logging.info('[1/8] Positive fingerprint, counts and formal-window inventory QC')
        positive=validate_positive(paths,cfg,report)
        inventory=inspect_inventory(paths['inventory'],cfg,folder,report)
        if not report['inventory_states_valid']:raise ValueError('Illegal/missing inventory states; see inventory/inventory_invalid_states.csv')
        logging.info('[2/8] Strict spatial alignment and historical buffer QC')
        layers=validate_spatial(paths,inventory,cfg,folder,report)
        logging.info('[3/8] Independently reconstructing candidates and reading existing scores')
        values=build_candidates(paths,layers,cfg,folder,report)
        logging.info('[4/8] Full-candidate quantiles and disjoint pools')
        lo,hi=define_pools(values,cfg,folder,report);del values
        logging.info('Actual thresholds: %.9f, %.9f; pools: %s',lo,hi,report['pool_counts'])
        logging.info('[5/8] Fixed-seed sampling without replacement')
        points=select_samples(paths,cfg,folder,report,lo,hi)
        logging.info('[6/8] Independent mask/score readback and selection QC')
        validate_selection(points,positive,paths,layers,cfg,folder,report,lo,hi)
        logging.info('[7/8] Distance, cross-strategy overlap and output consistency')
        points=distance_statistics(points,layers,cfg,folder,report)
        compare_overlap(points,comparisons,paths,folder)
        save_samples(points,cfg,folder,report)
        plot_samples(points,layers,cfg,folder)
        report['input_files_unchanged']=snapshot(inputs)==report['input_snapshot']
        if not report['input_files_unchanged']:raise ValueError('Input changed during run')
        report['final_negative_count']=len(points);report['qc_status']='PASS'
        logging.info('[8/8] QC PASS and writing final report')
    except Exception as exc:
        report['error']=str(exc);report['input_files_unchanged']=snapshot(inputs)==report['input_snapshot']
        logging.exception('NS3 sampling failed: %s',folder)
    finally:
        write_reports(cfg,report,folder)
        report['output_paths']=[str(p) for p in sorted(folder.rglob('*')) if p.is_file()]+[str(folder/'run_metadata.json')]
        write_json(folder/'run_metadata.json',report)
        logging.info('QC Status: %s; Output: %s',report['qc_status'],folder)
        for handler in logging.getLogger().handlers[:]:handler.close();logging.getLogger().removeHandler(handler)
    return folder,report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    try:
        _,report=run(args.config);return 0 if report['qc_status']=='PASS' else 1
    except Exception as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
