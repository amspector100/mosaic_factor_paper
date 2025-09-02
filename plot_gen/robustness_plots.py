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

from plot_utils import calc_mean_sem, most_recent_job_id, load_simulation_data, root_dir

mosaicperm_dir = root_dir.parent / "mosaicperm"
sys.path.insert(0, str(mosaicperm_dir))
import mosaicperm as mp
from mosaicperm.utilities import vrange, elapsed


def main():
    job_ids = [most_recent_job_id(sim_type='robustness_sims')]
    print(f"Using job id={job_ids}.")
    df = load_simulation_data(job_ids, sim_type='robustness_sims')
    df = df.loc[df['n'] == 300]
    df['sampling_method'] = df['sampling_method'].map(
        {
            'ar1': 'AR(1)',
            'garch': 'GARCH(1,1)',
            'garch_ar1': 'GARCH(1,1) + AR(1)',
            'mvn_arch': 'MVN-ARCH',
        }
    )
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
        + labs(x='Nominal level', y='Type I Error Rate', color='Industry', linetype='Industry')
    )
    g.save(f"{str(root_dir)}/plots/robustness_plot.png", dpi=500)



if __name__ == "__main__":
    main()