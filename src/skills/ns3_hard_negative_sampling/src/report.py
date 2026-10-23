import pandas as pd
from .utils import table,qname


def write_reports(cfg,report,run):
    lines=['# NS3 Multi-temporal Hard Negative Sampling','',f"QC Status: {report['qc_status']}",'',
        f"Positive数量：{report.get('positive_count','NA')}。源文件：{report.get('positive_source_path','NA')}",
        f"Positive SHA256：{report.get('positive_sha256','NA')}",
        f"正式时相：{', '.join(cfg['inventory']['temporal_fields'])}",
        f"Inventory数量：{report.get('inventory_polygon_count','NA')}；ever_landslide数量：{report.get('ever_landslide_count','NA')}。",
        f"非法状态：{report.get('invalid_state_count','NA')}；缺失状态：{report.get('missing_state_count','NA')}；组合状态次数：{report.get('combined_state_occurrences','NA')}。",
        f"时序警告：{report.get('sequence_warning_count','NA')}；{report.get('warning_types',{})}。警告不自动修改状态，窗口起点已有历史状态可能源于更早历史。",
        f"Buffer QC：{report.get('buffer_qc','NA')}",f"空间对齐：{report.get('spatial_alignment','NA')}",'']
    for file,title in [('candidate_pool_summary','候选池与实际分位数'),('sampling_summary','抽样数量和分数'),
        ('nearest_neighbor_qc','最近邻距离'),('historical_distance_summary','到历史滑坡及buffer距离'),('cross_strategy_overlap','跨策略同cell重叠')]:
        p=run/f'statistics/{file}.csv'
        if p.exists():lines += [f'## {title}','',table(pd.read_csv(p)),'']
    lines += [f"Positive同cell冲突数：{report.get('positive_same_cell_conflict_count','未执行')}。",
        f"重复row/col数：{report.get('duplicate_cell_count','未执行')}。",
        f"选中样本QC：{report.get('selection_qc','未执行')}。",
        f"GPKG/CSV一致性：{report.get('gpkg_csv_consistency','未执行')}。",'', '## WARNING','']
    lines += report.get('warnings',[]) or ['没有触发分位参考差异或配置的最近邻聚集警告。']
    if report['qc_status']=='FAIL':lines+=['','## 停止原因','',report.get('error','QC failure'),'失败run内任何中间结果不得作为正式样本使用。']
    lo=qname(cfg['hard_negative']['lower_quantile']);hi=qname(cfg['hard_negative']['upper_quantile'])
    lines += ['', '## 方法与限制','',f"Random来自score<{lo}；Hard来自{lo}<=score<{hi}；score>={hi}仅统计，不参加本次抽样。",
        f"Hard部分根据既有{cfg['inputs']['hardness_raster']['source_model']} score定义。score用于candidate ranking，不是校准后的滑坡发生概率。",
        'A不是永久稳定证据，U不是未发生证据，恢复对象仍排除。候选池是基于历史排除的操作性定义，不证明每个候选永久稳定。',
        'Random保持较广候选覆盖；高分排除是既定保守设计，不表示这些区域均错误。没有依据性能优化比例或阈值。',
        '每个pool等概率无放回抽样，无默认minimum spacing。最近邻和历史距离仅诊断，不据此删除或重新抽样。跨策略偶然同cell允许且未用于排除候选。',
        'NS3优劣需由后续统一RF实验检验；本次只生成负样本，不训练NS3_RF。Candidate Exploration不是运行前置依赖。']
    (run/'reports/ns3_sampling_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    qc=['# NS3 QC','',f"QC Status: {report['qc_status']}",'']
    qc += [f'- {k}: {v}' for k,v in report.items() if k not in ('input_snapshot','output_paths')]
    (run/'qc/qc_report.md').write_text('\n'.join(qc)+'\n',encoding='utf-8')
