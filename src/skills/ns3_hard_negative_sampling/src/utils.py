from datetime import datetime
from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd
from rasterio.windows import Window


def create_run(root, seed):
    run = root / (datetime.now().strftime('%Y%m%d_%H%M%S_%f')+f'_seed{seed}')
    run.mkdir(parents=True, exist_ok=False)
    for name in ('config', 'inventory', 'statistics', 'spatial', 'samples', 'figures', 'qc', 'reports', 'logs'):
        (run/name).mkdir()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', force=True,
        handlers=[logging.FileHandler(run/'logs/run.log', encoding='utf-8'), logging.StreamHandler()])
    return run


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def snapshot(paths):
    files = set()
    for path in paths.values():
        files.add(path)
        if path.suffix.lower() == '.shp':
            files.update(p for p in path.parent.glob(path.stem+'.*') if p.suffix.lower() in
                         ('.shp', '.shx', '.dbf', '.prj', '.cpg', '.sbn', '.sbx'))
    return {str(p): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in sorted(files)}


def windows(raster, rows):
    for r in range(0, raster.height, rows):
        yield Window(0, r, raster.width, min(rows, raster.height-r))


def valid(a):
    return ~np.ma.getmaskarray(a) & np.isfinite(a.data)


def profile(ref):
    return dict(driver='GTiff', width=ref.width, height=ref.height, count=1, crs=ref.crs,
        transform=ref.transform, dtype='uint8', nodata=255, tiled=True, compress='deflate', BIGTIFF='IF_SAFER')


def qname(q):
    return 'Q'+format(q*100, '.10g').replace('.', '_')


def table(frame):
    if frame.empty:
        return '无记录。'
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return format(v, '.8g') if np.isfinite(v) else 'NA'
        return str(v).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']+
        ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False, name=None)])
