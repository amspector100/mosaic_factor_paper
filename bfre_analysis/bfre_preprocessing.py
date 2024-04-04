"""
This file preprocesses the raw BFRE model data.
It saves the processed data into CACHE_DIR and includes
utility functions for loading the cached data.
"""

import os
import sys
# Import mosaicperm package---you can also just install mosaicperm via pip
sys.path.insert(0, "../../mosaicperm/")
import mosaicperm as mp
from mosaicperm.utilities import elapsed, vrange
sys.path.insert(0, "../")
from mosaic_paper_src import utilities, parser


# Typical imports
import time
import numpy as np
import scipy.sparse as sp
import pandas as pd
import datetime 
from tqdm.auto import tqdm

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from plotnine import *

DATA_DIR = "../../bfre_data"
CACHE_DIR = "../data/bfre_cache"
DEFAULT_NROWS = 100000

MANUAL_REMOVAL = ['LIBERTY BROADBAND SR.C', "LIBERTY MDA.SR.C LBRTY. SIRIUSXM", "DISCOVERY SERIES C"]
def remove_redundant_assets(assets, asset2names):
	"""
	Redundancies:
	e.g. remove "GOOGLE 'C'" if "GOOGLE A" is present
	"""
	anames = asset2names[assets].copy()
	anames = anames.str.replace("'", "")
	to_remove = []
	to_remove_inds = []
	for j, asset in enumerate(anames):
		if " " not in asset:
			continue
		end = asset.split(" ")[-1]
		if len(end) == 1:
			core = asset[:-1]
			alts = anames[(anames.str.contains(core)) & (anames != asset)]
			if np.any(np.array([asset > alt for alt in alts])):
				to_remove.append(anames.index[j])
				to_remove_inds.append(j)
				continue
		if asset in MANUAL_REMOVAL:
			to_remove.append(anames.index[j])
			to_remove_inds.append(j)
	return to_remove, to_remove_inds


def read_csv_progress_bar(file, chunksize=10, **kwargs):
	df_chunks = []
	for df_chunk in tqdm(pd.read_csv(file, chunksize=chunksize, **kwargs)):
		df_chunks.append(df_chunk)
	return pd.concat(df_chunks, axis=0)

def load_data(
	industry='',
	null_prop_thresh=0.0,
	cache_path=CACHE_DIR,
	which_factors='all',
	min_assets_per_industry=10,
	active_null_prop_thresh=0.95,
	start_date=datetime.datetime(2017, 1, 1),
	to_exclude=None,
):
	"""
	Parameters
	----------
	industry : str
		Specifies the industry to analyze.
	null_prop_thresh : int
		Only include assets if they are present more than this % of the time.
	cache_path : str
		Location of the cached data.
	factors : str
		one of 'all', 'industry', or 'style'
	min_assets_per_industry : int
		Minimum number of assets in the industry to be part of the "active set""
	active_null_prop_thresh : int
		Must be present this % of the time to be in the "active set".
	to_exclude : str
		Excludes assets whose industry contains this string
	"""
	# Read data
	industries = pd.read_csv(f"{CACHE_DIR}/industries.csv", index_col=0)['Industry']
	null_props = pd.read_csv(f"{CACHE_DIR}/null_proportions.csv", index_col=0)['null_prop']
	outcomes = pd.read_csv(f"{CACHE_DIR}/returns.csv", index_col=0)
	outcomes.index = pd.to_datetime(outcomes.index)
	# outcomes.index.map(lambda x: datetime.datetime(
	# 		year=int(str(x)[0:4]), month=int(str(x)[4:6]), day=int(str(x)[6:])
	# ))
	# glues indices and assets together
	aind = pd.Series(
		outcomes.columns, index=np.arange(len(outcomes.columns))
	)
	# read exposures
	exposures = np.load(f"{CACHE_DIR}/exposures.npy")
	factor_cols = np.load(f"{CACHE_DIR}/factor_cols.npy")

	## find desired subset of outcomes
	xinds = np.where(outcomes.index >= start_date)[0]
	exposures = exposures[xinds]
	outcomes = outcomes.iloc[xinds]

	### find desired subset of assets
	industry = str(industry).upper()
	if len(industry) > 0:
		assets = sorted(
			industries.index[industries.apply(lambda x: x[0:len(industry)] == industry)].tolist()
		)
	else:
		assets = industries.index.tolist()
	if to_exclude is not None:
		assets = [asset for asset in assets if to_exclude not in industries[asset]]
	assets = [asset for asset in assets if (1 - null_props[asset]) > null_prop_thresh]
	# indices for exposures
	indices = aind.index[aind.isin(assets)].values
	exposures = exposures[:, indices, :]
	

	### find desired subset of factors
	which_factors = str(which_factors).lower()
	if which_factors != 'all':
		ind_markers = industries.unique()
		if which_factors == 'industry':
			finds = [k for k, fc in enumerate(factor_cols) if fc in ind_markers or fc == 'MARKET']
		elif which_factors == 'style':
			finds = [k for k, fc in enumerate(factor_cols) if fc not in ind_markers or fc == 'MARKET']
		else:
			raise ValueError(f"Unrecognized value for which_factors={which_factors}")
		exposures = exposures[:, :, finds]
		factor_cols = factor_cols[finds]

	### get rid of redundant factors with missing/zero exposures
	non_redundant = np.mean(
		np.isnan(exposures) | (exposures == 0), axis=0
	).mean(axis=0) != 1
	factor_cols = factor_cols[non_redundant]
	exposures = exposures[:, :, non_redundant]

	### suggest active subset to increase power
	active_subset = []
	for i, asset in enumerate(assets):
		if np.sum(industries == industries[asset]) >= min_assets_per_industry:
			if 1 - null_props[asset] > active_null_prop_thresh:
				active_subset.append(i)
	active_subset = np.array(active_subset).astype(int)

	# Return output
	return dict(
		outcomes=outcomes[assets],
		exposures=exposures,
		industries=industries[assets],
		null_props=null_props[assets],
		factor_cols=factor_cols,
		active_subset=active_subset,
	)



def main(args):
	# Parse argument (nrows is helpful for debugging)
	args = parser.parse_args(args)
	nrows = args.get("nrows", [DEFAULT_NROWS])[0]
	print(f"Nrows = {nrows}.")

	## 1. load data
	t0 = time.time()
	data = []
	for ystart, yend in zip(
		['2013', '2018', '2021'], ['2017', '2020', '2023']
	):
		print(f"Loading data from {ystart}-{yend} at {elapsed(t0)}.")
		# df = read_csv_progress_bar(
		# 	file=f'{DATA_DIR}/bfre_factor_model_data_{ystart}_{yend}.csv',
		# 	chunksize=10,
		# 	header=[0,1],
		# 	index_col=0,
		# 	nrows=nrows,
		# )
		df = pd.read_csv(
			f'{DATA_DIR}/bfre_factor_model_data_{ystart}_{yend}.csv',
			header=[0,1], index_col=0, nrows=nrows,
		)
		data.append(df)
	print(f"Finished loading all data at {elapsed(t0)}.")
	data = pd.concat(data, axis='index')
	all_assets = np.sort(np.unique([x[0] for x in data.columns if x[0][0] == 'Z']))

	# remove redundant assets
	asset_names = pd.read_csv(f"{DATA_DIR}/assets_id_to_name.csv").rename(
		columns={"invariant_id":"ASSET", "name_sec":"name"}
	)
	asset2names = asset_names.set_index("ASSET")['name']
	names2asset = asset_names.set_index("name")['ASSET']
	to_remove, _ = remove_redundant_assets(all_assets, asset2names)
	all_assets = sorted(list(set(all_assets) - set(to_remove)))

	## 2. Factor columns and date parsing
	factor_cols = np.sort([c for c in data['Z913Y29A4'].columns if c not in ['EXRETURN', 'SRET', 'CAPT']])
	k = len(factor_cols)
	data.index = data.index.map(lambda x: datetime.datetime(
			year=int(str(x)[0:4]), month=int(str(x)[4:6]), day=int(str(x)[6:])
	))
	if nrows == DEFAULT_NROWS:
		np.save(f"{CACHE_DIR}/factor_cols.npy", factor_cols)

	## 3. Returns (outcomes)
	print(f"Creating returns at {elapsed(t0)}.")
	returns = data[[(asset, "EXRETURN") for asset in all_assets]].copy()
	returns.columns = returns.columns.get_level_values(0)
	returns.columns.name = 'ASSET'
	returns.index.name = 'Date'
	if nrows == DEFAULT_NROWS:
		returns.to_csv(f"{CACHE_DIR}/returns.csv")

	## 4. Create exposures
	print(f"Creating exposures at {elapsed(t0)}.")
	exposures = []
	for factor_col in factor_cols:
		exp = data[[(asset, factor_col) for asset in all_assets]].values
		exposures.append(exp)

	exposures = np.stack(exposures, axis=-1)
	if nrows == DEFAULT_NROWS:
		np.save(f"{CACHE_DIR}/exposures.npy", exposures)

	## 5. Industry markers
	if nrows == DEFAULT_NROWS:
		print(f"Creating industries at {elapsed(t0)}.")
	exp2 = exposures.copy()
	exp2[np.isnan(exp2)] = 0
	prop_int = np.mean(exp2.astype(int) == exp2.astype(float), axis=0).mean(axis=0)
	ind_markers = set(factor_cols[prop_int == 1].tolist())
	ind_markers -= set(['MARKET', 'USD'])
	ims = sorted(list(ind_markers))
	ind_means = data[all_assets].fillna(0).mean(axis=0).unstack()
	industries = ind_means[ims].idxmax(axis=1)
	industries.name = 'Industry'
	if nrows == DEFAULT_NROWS:
		industries.to_csv(f"{CACHE_DIR}/industries.csv")

	## 6. Null proportions
	print(f"Creating null_props at {elapsed(t0)}.")
	null_props = np.zeros(len(all_assets))
	for i, asset in enumerate(all_assets):
		null_props[i] = np.mean(
			np.any(np.isnan(exposures[:, i]), axis=-1) | 
			returns[asset].isnull().values
		)
	null_props = pd.Series(null_props, index=all_assets)
	null_props.index.name = 'ASSET'
	null_props.name = 'null_prop'
	if nrows == DEFAULT_NROWS:
		null_props.to_csv(f"{CACHE_DIR}/null_proportions.csv")

	## 7. Exposure subset for simulations
	if nrows == DEFAULT_NROWS:
		target = datetime.datetime(year=2020, month=4, day=17)
	else:
		target = datetime.datetime(year=2018, month=1, day=2)
	ind = np.where(returns.index == target)[0].item()
	sim_exposures = exposures[ind].copy()
	# subset to financial assets
	fin_inds = np.array([
		i for i, asset in enumerate(all_assets) 
		if industries[asset][0:3] == 'FIN'
	])

	# get rid of assets which have all nan exposures
	sim_exposures = sim_exposures[fin_inds].copy()
	sim_exposures = sim_exposures[~np.all(np.isnan(sim_exposures), axis=1)].copy()
	sim_exposures[np.isnan(sim_exposures)] = 0
	sim_exposures = sim_exposures[:, ~np.all(sim_exposures == 0, axis=0)]
	if nrows == DEFAULT_NROWS:
		np.save(f"{CACHE_DIR}/simulation_exposures.npy", sim_exposures)


if __name__ == '__main__':
	main(sys.argv)
