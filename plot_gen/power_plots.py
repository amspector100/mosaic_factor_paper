import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from plotnine import *
import glob

import os
import sys
import time

from plot_utils import calc_mean_sem, most_recent_job_id, load_simulation_data, root_dir
sys.path.append(str(root_dir / "sims"))
from main_sims import load_L_matrix, sample_data

def compute_r2(n, seed, rho, sparsity, L, eps_dist):
    data = sample_data(n, seed, rho, sparsity, L, eps_dist)
    signal = rho * data['Z'].reshape(-1, 1) * data['v'].reshape(1, -1)
    return np.mean(signal**2) / np.mean(data['eps']**2)


def main():
    df = load_simulation_data(
        job_ids=[most_recent_job_id(sim_type='main_sims')],
        sim_type='main_sims'
    )
    alpha = 0.05

    ## OLS thresholds
    ols_sub = df.loc[df['method'].str.contains("OLS")].copy()
    group_vals = ['sparsity', 'test_stat', 'test_stat_index', 'method', 'n', 'eps_dist']
    thresh = ols_sub.loc[ols_sub['rho'] == 0].groupby(group_vals)['T'].quantile(1-alpha)
    thresh = thresh.reset_index().rename(columns={"T":"threshold"})
    ols_sub = pd.merge(
        ols_sub, thresh, on=group_vals, how='left'
    )
    ols_sub['disc'] = ols_sub['T'] > ols_sub['threshold']
    ## Mosaic results
    mpt_sub = df.loc[~df['method'].str.contains("OLS")].copy()
    mpt_sub['disc'] = (mpt_sub['pval'] <= alpha).astype(float)
    ## final df
    fdf = pd.concat([mpt_sub, ols_sub], axis='index').drop("threshold", axis='columns')
    ## aggregate statistics
    agg = calc_mean_sem(
        fdf,
        group_vals=group_vals+['rho'],
        meas=['disc']
    )

    ## adaptive
    mpt_adaptive = agg.loc[
        (agg['method'] == 'MPT') 
        & (agg['test_stat_index'] == 'adaptive')
    ]
    # A-priori maximum
    nonadapt = agg.loc[agg['test_stat_index'] != 'adaptive']
    ids = nonadapt.groupby(
        list(set(group_vals+['rho']) - set(['test_stat_index']))
    )['disc_mean'].idxmax().values
    oracle_index_stats = nonadapt.loc[ids]
    oracle_index_stats['test_stat_index'] = 'oracle'
    ## prepare for plotting
    df4plot = pd.concat([oracle_index_stats, mpt_adaptive], axis='index')
    df4plot['Method'] = (df4plot['method'] + df4plot['test_stat_index']).map({
        "MPTadaptive":"MPT (adaptive)",
        "MPToracle":"MPT (oracle)",
        "OLS oracleoracle":r"OLS (double oracle)"
    })
    ## comparisons---this output is used in the paper main text
    comparisons = []
    for n in df4plot['n'].unique():
        for eps_dist in df4plot['eps_dist'].unique():
            for sparsity in df4plot['sparsity'].unique():
                # Consider one value of sparsity and n
                sub = df4plot.loc[
                    (df4plot['sparsity'] == sparsity) &
                    (df4plot['n'] == n) &
                    (df4plot['eps_dist'] == eps_dist)
                ]
                if len(sub) == 0:
                    continue
                for rho in sub['rho'].unique():
                    subrho = sub.loc[sub['rho'] == rho]
                    # For this value of rho, find the power of the two methods with the same test statistic
                    power_mpt = subrho.loc[subrho['Method'] == 'MPT (oracle)', 'disc_mean'].item()
                    power_ols = subrho.loc[subrho['Method'] == 'OLS (double oracle)', 'disc_mean'].item()
                    # find minimum value of rho where MPT (oracle) exceeds OLS
                    minrho = sub.loc[
                        (sub['Method'] == 'MPT (oracle)') &
                        (sub['disc_mean'] >= power_ols),
                        'rho'
                    ].min()
                    comparisons.append([n, sparsity, eps_dist, rho, power_ols, power_mpt, minrho])
    comparisons = pd.DataFrame(comparisons, columns=['n', 'sparsity', 'eps_dist', 'rho', 'power_ols', 'power_mpt', 'minrho'])
    ## Display
    comparisons['power_diff'] = comparisons['power_ols'] - comparisons['power_mpt']
    #comparisons['rho_ratio'] = comparisons['minrho'] / comparisons['rho']
    print(comparisons.groupby(['n', 'sparsity', 'eps_dist'])['power_diff'].max())
    print(f"Average power difference is {comparisons['power_diff'].mean()}")
    ### Convert rho to an interpretable metric
    L = load_L_matrix()
    unique_vals = df4plot[['rho', 'sparsity', 'eps_dist']].drop_duplicates()
    unique_vals['r2'] = unique_vals.apply(lambda row: compute_r2(n=1000, seed=1, rho=row['rho'], sparsity=row['sparsity'], L=L, eps_dist=row['eps_dist']), axis=1)
    df4plot = pd.merge(df4plot, unique_vals, on=['rho', 'sparsity', 'eps_dist'], how='left')
    ### Plot
    meas = 'disc'
    g = (
        ggplot(
            df4plot.loc[
                (df4plot['test_stat'] == 'quant_corr')],
            aes(x='r2', y=f'{meas}_mean', color='Method')
        ) 
        + geom_point(size=0.5)
        + geom_line()
        + geom_errorbar(aes(ymin=meas+"_ymin", ymax=meas+"_ymax"), width=0.001)
        + facet_wrap("~sparsity", labeller=lambda x: rf"$s_0$={x}", nrow=1)
        + theme_bw()
        + theme(figure_size=(8,3))
        + geom_hline(yintercept=alpha, color='black', linetype='dotted')
        + scale_color_manual(['blue', 'red', 'black'])
        + labs(
            #x=r'Signal size ($\rho$)', 
            x=r'$\mathbb{E}[\|Z \cdot v\|^2] / \mathbb{E}[\|\epsilon\|^2]$',
            y='Power', 
        )
    )
    g.save(root_dir / "plots" / "power_plot.png", dpi=500)

if __name__ == "__main__":
    main()