import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from pathlib import Path
import argparse
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

@dataclass
class HistConfig:
    label: str
    color: str

def create_naive_methods_plot(data: pd.DataFrame):
    ## Main plot
    data = data.loc[data['center_date'] == 'covid']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))

    # The bootstrap suggested by our reviewer
    regular_bs = 'bootstrap_residualizeTrue_methoddefault_quantileTrue'
    olslab = 'OLS statistic, S($\hat\epsilon^{OLS}$)'


    COLOR_DICT = {
        "MPT":{
            "statistic":HistConfig(label="Mosaic statistic, S($\hat\epsilon$)", color="cornflowerblue"), 
            "null_stat":HistConfig(label="Mosaic permutations", color="orangered")
        },
        regular_bs:{
            "statistic":HistConfig(label=olslab, color="blue"), 
            "null_stat":HistConfig(label="Bootstrap distribution", color="green")
        },
        "Naive Permutation":{
            "statistic":HistConfig(label=olslab, color="blue"), 
            "null_stat":HistConfig(label="Naive perm. test", color="gray")
        },
    }

    for axnum, method in zip(
        [0, 1, 2],
        ['Naive Permutation', regular_bs, 'MPT'],
    ):
        ax = axes[axnum]
        sub = data.loc[data['method'] == method]
        for col in ['null_stat', 'statistic']:
            try:
                histconfig = COLOR_DICT[method][col]
            except KeyError:
                # this signals that we shouldn't plot this
                continue
            sns.histplot(
                sub[col].values,
                color=histconfig.color,
                alpha=0.5, 
                ax=ax, 
                label=histconfig.label,
                linewidth=0.2,
                #bins=mondbins if 'Mosaic' in label else 12,
            )
        ax.legend()

        if axnum != 0:
            axes[axnum].set(ylabel='')
        axes[axnum].set_ylim(0, 240)
        
    for axchar, ax in zip(['a', 'b', 'c'], axes):
        ax.set(title=f"({axchar})")
    for ax in axes:
        ax.legend()

    os.makedirs(root_dir / "plots/", exist_ok=True)
    plt.savefig(root_dir / "plots" / "naive_methods.png", dpi=500, bbox_inches='tight')
    plt.close()
    #plt.show()

def appendix_bootstrap_plot(data: pd.DataFrame):
    dates = data['center_date'].unique()
    fig, axes = plt.subplots(len(dates), 5, figsize=(12, len(dates)*3))
    olslab = 'OLS statistic, S($\hat\epsilon^{OLS}$)'
    method_names = {
        "bootstrap_residualizeTrue_methodblock_quantileTrue":"Block bootstrap",
        "bootstrap_residualizeTrue_methoddefault_quantileTrue":"Bootstrap",
        "bootstrap_residualizeTrue_methodwithin_block_quantileTrue":"Within-block\nbootstrap",
        "Naive Permutation":"Naive Permutation",
        "MPT":"MPT",
    }
    # config
    COLOR_DICT = {
        "MPT":{
            "statistic":HistConfig(label="Mosaic statistic, S($\hat\epsilon$)", color="cornflowerblue"), 
            "null_stat":HistConfig(label="Mosaic permutations", color="orangered")
        },
        "Naive Permutation":{
            "statistic":HistConfig(label=olslab, color="blue"), 
            "null_stat":HistConfig(label="Naive perm. test", color="gray")
        },
    }
    for method in method_names.keys():
        if 'bootstrap' in method_names[method].lower():
            COLOR_DICT[method] = {
                "statistic":HistConfig(label=olslab, color="blue"), 
                "null_stat":HistConfig(label=method_names[method], color="green")
            }
    # plot
    for rownum, date in enumerate(dates):
        for axnum, method in enumerate(method_names):
            ax = axes[rownum][axnum]
            sub = data.loc[(data['method'] == method) & (data['center_date'] == date)]
            for col in ['null_stat', 'statistic']:
                histconfig = COLOR_DICT[method][col]
                sns.histplot(
                    sub[col].values,
                    color=histconfig.color,
                    alpha=0.5, 
                    ax=ax, 
                    linewidth=0.2,
                )
            if rownum == 0:
                ax.set(title=method_names[method])

            if axnum != 0:
                axes[rownum][axnum].set(ylabel='')
            if axnum == 0:
                axes[rownum][axnum].set(ylabel=f'Date={date.capitalize()}\n\nCount')
            axes[rownum][axnum].set_ylim(0, 240)
            
    plt.subplots_adjust(wspace=0.28)
    os.makedirs(root_dir / "plots/", exist_ok=True)
    plt.savefig(root_dir / "plots" / "appendix_bootstrap_plot.png", dpi=500, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, default='none')
    args = parser.parse_args(sys.argv[1:])
    if args.job_id == 'none':
        job_id = most_recent_job_id(sim_type='bootstrap_sims')
    else:
        job_id = args.job_id
    ## Load data
    data = load_simulation_data(
        job_ids=[job_id],
        sim_type='bootstrap_sims'
    )
    ## Plot
    create_naive_methods_plot(data)    
    appendix_bootstrap_plot(data)

if __name__ == "__main__":
    main()