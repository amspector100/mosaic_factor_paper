import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from plotnine import *
import glob

import os
import sys
import time

root_dir = Path(__file__).parent.parent
mosaicperm_dir = root_dir.parent / "mosaicperm"
sys.path.insert(0, str(mosaicperm_dir))
import mosaicperm as mp
from mosaicperm.utilities import vrange, elapsed

def calc_mean_sem(data, group_vals, meas, trunc_zero=True):
    """
    Groups data by group_vals and then calculates mean, standard error
    for each column.
    """
    agg_df = data.groupby(group_vals)[meas].agg(['mean', 'std', 'sem']).reset_index()
    for m in meas:
        agg_df[f'{m}_mean'] = agg_df[m]['mean']
        agg_df[f'{m}_std'] = agg_df[m]['std']
        agg_df[f'{m}_sem'] = agg_df[m]['sem']
        agg_df[f'{m}_ymin'] =  agg_df[f'{m}_mean'] - 2*agg_df[f'{m}_sem']
        if trunc_zero:
            agg_df[f'{m}_ymin'] = np.maximum(0, agg_df[f'{m}_ymin'])
        agg_df[f'{m}_ymax'] =  agg_df[f'{m}_mean'] + 2*agg_df[f'{m}_sem']
    
    agg_df = agg_df.loc[:, agg_df.columns.get_level_values(1) == '']
    agg_df.columns = agg_df.columns.get_level_values(0)
    return agg_df

def most_recent_job_id():
    fnames = glob.glob(f"{str(root_dir)}/sim_data/robustness_sims/*/*/*.csv")
    fnames = [Path(fname).stem.split("id")[-1].split("_")[0] for fname in fnames]
    return int(max(fnames))


def load_simulation_data(
    job_ids: list[int],
):
    data = []
    for job_id in job_ids:
        fnames = glob.glob(f"{str(root_dir)}/sim_data/robustness_sims/*/*/*{job_id}*.csv")
        for fname in fnames:
            data.append(pd.read_csv(fname))
    return pd.concat(data, axis='index')

def main():
    job_ids = [most_recent_job_id()]
    df = load_simulation_data(job_ids)
    df = df.loc[df['n'] == 300]
    # create new df
    new_df = []
    alphas = np.linspace(0.01, 0.2, 10)
    for alpha in alphas:
        sub = df.copy()
        sub['disc'] = sub['pval'] <= alpha
        sub['alpha'] = alpha
        new_df.append(sub)
    new_df = pd.concat(new_df, axis='index')
    agg = calc_mean_sem(
        new_df,
        group_vals=['alpha', 'industry', 'sampling_method'],
        meas=['disc']
    )
    # add xeqy
    xeqy = agg.loc[agg['industry'] == 'EGY'].copy()
    xeqy['industry'] = 'x=y'
    xeqy['disc_mean'] = xeqy['alpha']
    xeqy['disc_ymin'] = xeqy['alpha']
    xeqy['disc_ymax'] = xeqy['alpha']
    agg_full = pd.concat([agg, xeqy], axis='index')
    # plot
    g = (
        ggplot(agg_full, aes(x='alpha', y='disc_mean', color='industry', linetype='industry'))
        + geom_line()
        + facet_wrap('~sampling_method', ncol=4)
        + geom_errorbar(aes(ymin='disc_ymin', ymax='disc_ymax'), width=0.001)
        + theme_bw()
        + geom_point(data=agg)
        + theme(figure_size=(8, 4))
        + scale_color_manual(['red', 'blue', 'green', 'black'])
        + scale_linetype_manual(['solid', 'solid', 'solid', 'dotted'])
    )
    g.save(f"{str(root_dir)}/plots/robustness_plot.png", dpi=500)



if __name__ == "__main__":
    main()