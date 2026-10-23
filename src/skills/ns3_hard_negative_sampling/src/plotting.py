import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_samples(points,layers,cfg,run):
    dpi=cfg['plots']['dpi'];groups=[k for k in ('random','hard') if (points.negative_source==k).any()]
    fig,ax=plt.subplots(figsize=(8,5),facecolor='white')
    for key in groups:
        v=np.sort(points.loc[points.negative_source==key,'hardness_score'].to_numpy())
        ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',label=key)
    ax.set(xlabel='Hardness (candidate ranking score)',ylabel='ECDF');ax.legend();fig.tight_layout()
    fig.savefig(run/'figures/selected_hardness_ecdf.png',dpi=dpi);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5),facecolor='white')
    ax.boxplot([points.loc[points.negative_source==key,'hardness_score'] for key in groups],tick_labels=groups)
    ax.set(ylabel='Hardness (candidate ranking score)');fig.tight_layout()
    fig.savefig(run/'figures/selected_hardness_boxplot.png',dpi=dpi);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,8),facecolor='white')
    layers['boundary'].boundary.plot(ax=ax,linewidth=.6)
    for key in groups:
        part=points[points.negative_source==key];ax.scatter(part.x,part.y,s=9,label=key,alpha=.7)
    ax.set(xlabel='Easting (CRS units)',ylabel='Northing (CRS units)',aspect='equal');ax.legend();fig.tight_layout()
    fig.savefig(run/'figures/ns3_sample_spatial_distribution.png',dpi=dpi);plt.close(fig)
