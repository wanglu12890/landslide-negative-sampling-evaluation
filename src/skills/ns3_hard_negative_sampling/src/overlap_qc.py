import pandas as pd
import rasterio
from .input_validation import read_points
from .utils import csv


def compare_overlap(points,comparisons,paths,run):
    rows=[]
    with rasterio.open(paths['reference_raster']) as ref:
        for name,path in comparisons.items():
            other=read_points(path,ref)
            cells=set(zip(other.row,other.col))
            count=sum((r,c) in cells for r,c in zip(points.row,points.col))
            rows.append(dict(comparison_strategy=name,same_cell_count=count,same_cell_ratio_of_NS3=count/len(points)))
    csv(pd.DataFrame(rows,columns=['comparison_strategy','same_cell_count','same_cell_ratio_of_NS3']),run/'statistics/cross_strategy_overlap.csv')
