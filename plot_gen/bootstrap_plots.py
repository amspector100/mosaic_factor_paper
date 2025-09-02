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

@dataclass
class HistConfig:
    label: str
    color: str

def create_naive_methods_plot(data: pd.DataFrame):
    ## Main 
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

    for axnum, methods in zip(
        [0, 1, 2],
        [['Naive Permutation'], [regular_bs], ['MPT']],
    ):
        ax = axes[axnum]
        for method in methods:
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

def main():
    ## Load data
    data = load_simulation_data(
        job_ids=[most_recent_job_id(sim_type='bootstrap_sims')],
        sim_type='bootstrap_sims'
    )
    ## Plot
    create_naive_methods_plot(data)    

if __name__ == "__main__":
    main()