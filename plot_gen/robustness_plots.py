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
from argparse import ArgumentParser
from plot_utils import calc_mean_sem, most_recent_job_id, load_simulation_data, root_dir

sys.path.insert(0, str(root_dir))
from mosaic_paper_src import loading, nonexch_sampling
import mosaicperm as mp # context in mosaic_paper_src will take care of this
sys.path.insert(0, str(root_dir / "bfre_analysis/"))
from bfre_preprocessing import load_data

import pdb
# mosaicperm_dir = root_dir.parent / "mosaicperm"
# sys.path.insert(0, str(mosaicperm_dir))
# import mosaicperm as mp
# from mosaicperm.utilities import vrange, elapsed

PLACEHOLDER_DATA_PATH = Path(root_dir) / "data" / "bfre_placeholder"

def main_robustness_plots():
    job_ids = [most_recent_job_id(sim_type='robustness_sims')]
    print(f"Using job id={job_ids}.")
    df = load_simulation_data(job_ids, sim_type='robustness_sims')
    df = df.loc[df['n'] == 300]
    df['sampling_method'] = df['sampling_method'].map(
        {
            'ar1': 'AR(1)',
            'garch': 'GARCH(3,3)',
            'garch_ar1': 'GARCH(3,3) + AR(1)',
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
        + theme(figure_size=(8, 3))
        + scale_color_manual(['red', 'blue', 'green', 'black'])
        + scale_linetype_manual(['solid', 'solid', 'solid', 'dotted'])
        + labs(x='Nominal level', y='Rejection probability', color='Industry', linetype='Industry')
    )
    # make sure there's enough space between panels
    g += theme(panel_spacing=0.026)
    g.save(f"{str(root_dir)}/plots/robustness_plot.png", dpi=500)
    
def parameter_distribution_plots():
    print("="*50)
    print("Plotting parameter distributions.")
    print("="*50)
    t0 = time.time()
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for indnum, industry in enumerate(['EGY', 'FIN', 'HLC']):
        data = load_data(industry=industry, use_placeholder=False)
        # outcomes, exposures
        outcomes = data['outcomes'].fillna(0).values
        T, N = outcomes.shape
        exposures = data['exposures']
        exposures[np.isnan(exposures)] = 0
        # residuals
        print(f"Fitting residuals for industry={industry}.")
        residuals = mp.factor.ols_residuals(outcomes=outcomes, exposures=exposures)

        # load params
        garch_params = loading.load_garch_params(industry=industry)
        mvn_arch_params = loading.load_mvn_arch_params(industry=industry)
        # compute r^2
        garch_r2s = np.zeros(N)
        arch_r2s = np.zeros(N)
        for i in tqdm(range(N)):
            tseries = outcomes[:, i]
            residuals = outcomes[tseries != 0]
            tseries = tseries[tseries != 0]
            if len(tseries) < 30:
                continue
            garch_sigma2s = np.zeros(len(tseries)-1)
            # compute garch sigma2s
            omega_garch = garch_params[f"omega"].values[i]
            for t in range(1, len(tseries)):
                for k in range(3):
                    if t-k >= 0:
                        garch_sigma2s[t-1] += garch_params[f"beta{k+1}"].values[i] * residuals[t-k, i]**2 
                        garch_sigma2s[t-1] += garch_params[f"alpha{k+1}"].values[i] * garch_sigma2s[t-k-1]
                garch_sigma2s[t-1] += omega_garch
            # compute arch sigma2s
            omega_arch = mvn_arch_params.values[i, 0]
            alpha_arch = mvn_arch_params.values[i, 1:]
            arch_sigma2s = (residuals**2) @ alpha_arch + omega_arch
            # compute r2s
            garch_r2s[i] = np.sum((garch_sigma2s-omega_garch)**2) / np.sum(garch_sigma2s**2)
            arch_r2s[i] = np.sum((arch_sigma2s-omega_arch)**2) / np.sum(arch_sigma2s**2)

        # plot autocorrelation estimates
        sns.histplot(garch_params['rho'].values, color='cornflowerblue', ax=axes[indnum][0])
        sns.histplot(garch_r2s, color='cornflowerblue', ax=axes[indnum][1])
        sns.histplot(arch_r2s, color='cornflowerblue', ax=axes[indnum][2])

        # add vertical mean lines and annotations
        hist_data_and_axes = [
            (garch_params['rho'].values, axes[indnum][0]),
            (garch_r2s, axes[indnum][1]),
            (arch_r2s, axes[indnum][2]),
        ]
        for data_arr, ax in hist_data_and_axes:
            mean_val = float(np.nanmean(data_arr))
            sd = float(np.nanstd(data_arr))
            #ax.axvline(mean_val, color='black', linestyle='--', linewidth=1.2)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax + 0.11 * (ymax - ymin))
            xmin, xmax = ax.get_xlim()
            ax.text(
                (xmax + xmin) / 2,
                ymin + 1.06 * (ymax - ymin),
                f"Mean={mean_val:.2f}, SD={sd:.2f}.",
                #rotation=90,
                ha='center',
                va='top',
                color='black',
            )
        # titles
        if indnum == 0:
            axes[indnum][0].set_title('Autocorrelation')
            axes[indnum][1].set_title(r'GARCH $R^2$')
            axes[indnum][2].set_title(r'MVN-ARCH $R^2$')
        else:
            axes[indnum][0].set_title('')
            axes[indnum][1].set_title('')
            axes[indnum][2].set_title('')
        axes[indnum][0].set_ylabel(f'{industry}\n\nCount')
        axes[indnum][1].set_ylabel('')
        axes[indnum][2].set_ylabel('')

    plt.subplots_adjust(wspace=0.25)
    plt.savefig(f"{str(root_dir)}/plots/estimated_sim_params.png", dpi=500)
    plt.close()

def main(args):
    parser = ArgumentParser()
    parser.add_argument("--skip_main_plot", action="store_true")
    args = parser.parse_args(args)
    if not args.skip_main_plot:
        main_robustness_plots()
    parameter_distribution_plots()


if __name__ == "__main__":
    main(sys.argv[1:])