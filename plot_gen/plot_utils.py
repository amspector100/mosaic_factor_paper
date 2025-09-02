import numpy as np
import pandas as pd
import glob
from pathlib import Path

root_dir = Path(__file__).parent.parent

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

def most_recent_job_id(sim_type: str):
    fnames = glob.glob(f"{str(root_dir)}/sim_data/{sim_type}/*/*/*.csv")
    fnames = [Path(fname).stem.split("id")[-1].split("_")[0] for fname in fnames]
    return int(max(fnames))


def load_simulation_data(
    job_ids: list[int],
    sim_type: str,
):
    data = []
    for job_id in job_ids:
        fnames = glob.glob(f"{str(root_dir)}/sim_data/{sim_type}/*/*/*{job_id}*.csv")
        for fname in fnames:
            data.append(pd.read_csv(fname))
    return pd.concat(data, axis='index')
