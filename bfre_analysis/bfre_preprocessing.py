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
from typing import Optional

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from plotnine import *

DATA_DIR = "../data/bfre"
CACHE_DIR = "../data/bfre_cache"
PLACEHOLDER_DIR = "../data/bfre_placeholder"
DEFAULT_NROWS = None # useful for debugging since reading the raw data takes ~15 min

###############################################
## This handles missing assets appropriately ##
###############################################
def compute_active_subset(
	residuals: np.array, 
	subset: Optional[np.array]=None,
	thresh: float=0.0,
):
	"""
	Discards residuals which are missing.
	(In the main analysis this is handled
	by providing the relevant input ``subset" but
	this function is used in the R^2 analysis.)
	"""
	if subset is None:
		subset = np.arange(residuals.shape[1])
	# discard assets which are missing >thresh% of the time
	nan_props = np.mean(np.isnan(residuals), axis=0)
	subset = set(subset.tolist()).intersection(
		list(set(np.where(nan_props <= thresh)[0]))
	)
	subset = subset.intersection(
		set(list(np.where(np.nanstd(residuals, axis=0) > 0)[0]))
	)
	return np.array(list(subset)).astype(int)

#####################
### PREPROCESSING ###
#####################
# most redundancies are handled automatically, but these are coded by hand
REDUNDANT = [
	### HLC
	'Z915M89N1', # BIO-RAD LABS B, duplicate of Z913Y5PL5 (BIO-RAD LABORATORIES 'A')
	### TECH
	'Z913Y29A4', # DISCOVERY SERIES C; duplicate of Z915NQMN5 (WARNER BROS DISCOVERY SERIES A)
	'Z91NAZR05', # LIBERTY BROADBAND SR.C; duplicate of Z91NAZQV8 (LIBERTY BROADBAND SR.A)
	'Z91ZXUML3', # LIBERTY MDA.SR.C LBRTY. SIRIUSXM; duplicate of Z91ZXUEU2 (LIBERTY MDA.SR.A LBRTY. SIRIUSXM)
	'Z95WJV2F5', # MOBILEYE GLOBAL A, duplicate of Z91LSL9K7 (MOBILEYE)
	'Z94V0ST82', # PERSHING SQUARE TONTINE HOLDINGS UNITS, duplicate of Z94Y26BU2 (PERSHING SQUARE TONTINE HOLDINGS A)
	### ENERGY
	'Z915MB219', # ALLIANCE RSO.PTNS.L P UT LP.; duplicate of Z915N9CG9 (ALLIANCE HOLDINGS GP)
	'Z915MAK78', # DENBURY RES., duplicate of Z94YJU272 (DENBURY)
	'Z923V4623', # EXTRACTION OIL &.GAS, duplicate of Z956QDZ27 (EXTRACTION OIL GAS)
	'Z915PDZ08', # C&J ENERGY SERVICES, duplicate of Z926JN0W8 (C&J ENERGY SVS.)
	'Z91LU5J10', # TRANSOCEAN PARTNERS, duplicate of Z917GPF17 (TRANSOCEAN)
	'Z913Y29F3', # KINDER MORGAN MAN., duplicate of Z915NT1G7 (KINDER MORGAN)
	# Below is not exactly a duplicate but nonetheless not an interesting alternative
	'Z9155FTD4', # MARATHON PETROLEUM, correlated with Z915MEDH6 (MARATHON OIL)
	### FIN
	'Z9196JFY6', # HEALTHCARE REALTY TRUST A, duplicate of Z913Y5LD7 (HEALTHCARE REAL.TST.)
	## CDI
	'Z913Y2WC4', # COMCAST SPECIAL 'A', duplicate of Z915M7ZV6 (COMCAST A)
	'Z915NSRS5', # SPECTRUM BRAND HOLDINGS, duplicate of  Z915NYMK4 (SPECTRUM BRANDS HOLDINGS)
	## IND 
	'Z915M99B5', # HEICO NEW 'A'; duplicate of Z913Y5LG0 (HEICO)
	'Z95CZAPF2', # HERTZ GLOBAL HLDGS; duplicate of Z9219VF89 (HERTZ GLOBAL HOLDINGS)
]
def remove_redundant_assets(assets, asset2names, outcomes, industries):
	"""
	Removes duplicate assets which overlap. 
	Also removes "GOOGLE 'C'" if "GOOGLE A" is present.
	"""
	anames = asset2names[assets].copy()
	to_remove = []
	to_remove_inds = []
	# exact duplicates
	nuq = np.unique(anames)
	to_drop = np.zeros(len(anames), dtype=bool)
	for name in nuq:
		flags = anames == name
		inds = np.where(flags)[0]
		if np.sum(flags) > 1:
			# Check for overlap; if overlap, drop asset with least missing data
			missing_pattern = np.isnan(outcomes)[:, inds]
			if np.any((~missing_pattern).sum(axis=1) > 1):
				nprops = missing_pattern.mean(axis=0)
				to_drop[flags] = True
				to_drop[inds[np.argmin(nprops)]] = False
	to_remove.extend(anames.index[to_drop].tolist())
	to_remove_inds.extend(np.where(to_drop)[0])
	print(f"Removed {len(to_remove)} exact duplicates.")
	# Remove Google 'C' if Google A is present.
	# Parsing so that 'A' becomes A, CL.A becomes A, SR. becomes A, etc.
	anames = anames.str.replace("'", "", regex=False)
	anames = anames.str.replace("CL.", " ", regex=False)
	anames = anames.str.replace("SR.", " ", regex=False)
	for j, (asset_code, asset_name) in enumerate(zip(assets, anames)):
		if j in to_remove_inds and asset_code in to_remove:
			continue
		end = asset_name.split(" ")[-1]
		if len(end) == 1 and len(asset_name) > 1:
			core = asset_name[:-2] # core name of the asset after removing " A" or " B"
			# Check if other assets have the same core name; if so deduplicate
			alts = anames[(anames.str.contains(core)) & (anames != asset_name)]
			if np.any(np.array([asset_name >= alt for alt in alts])):
				to_remove.append(anames.index[j])
				to_remove_inds.append(j)
				continue
		# manual entries
		if asset_code in REDUNDANT:
			to_remove.append(anames.index[j])
			to_remove_inds.append(j)
			continue
		# remove partner (shell) corps from EGYOGINT and FIN
		# don't do this in other industries; e.g., in healthcare,
		# "Surgery Partners" is not a shell corp. (see paper for details)
		patterns = ['PARTNER', 'PTN', 'UNIT']
		if industries[asset_code] in ['EGYOGINT', 'FIN']:
			if np.any([y in asset_name for y in patterns]):
				to_remove.append(anames.index[j])
				to_remove_inds.append(j)
				continue
	return to_remove, to_remove_inds

def load_data(
	industry='',
	which_factors='all',
	start_date=datetime.datetime(2017, 1, 1),
	to_exclude=None,
	cache_dir=None,
	use_placeholder=False,
):
	"""
	Parameters
	----------
	industry : str | list
		Specifies the industry/industries to analyze.
	which_factors : str
		one of 'all', 'industry', or 'style'
	start_date : datetime.datetime
		Starting date of the analysis.
	to_exclude : str | list
		Excludes assets whose industry contains this string
	cache_dir : str
		Location of the cached data
	use_placeholder : bool
		If True, uses publicly available placeholder data.
	"""
	# Default directories to load data
	if cache_dir is None and use_placeholder:
		cache_dir = PLACEHOLDER_DIR
	if cache_dir is None and not use_placeholder:
		cache_dir = CACHE_DIR

	# Read data
	industries = pd.read_csv(f"{cache_dir}/industries.csv", index_col=0)['Industry']
	outcomes = pd.read_csv(f"{cache_dir}/returns.csv", index_col=0)
	outcomes.index = pd.to_datetime(outcomes.index)
	outcomes.columns = outcomes.columns.astype(str)

	# glues indices and assets together
	aind = pd.Series(
		outcomes.columns, index=np.arange(len(outcomes.columns))
	)
	# read exposures (low memory read since we only use a subset later)
	exposures = np.load(f"{cache_dir}/exposures.npy", mmap_mode='r+')
	factor_cols = np.load(f"{cache_dir}/factor_cols.npy")

	## find desired subset of timepoints
	xinds = np.where(outcomes.index >= start_date)[0]
	exposures = exposures[xinds]
	outcomes = outcomes.iloc[xinds]

	### find desired subset of assets
	# Subset by industry (allowing multiple industries)
	if isinstance(industry, str):
		industry = [industry]
	industry = [str(x).upper() for x in industry]
	# If industry=[''], use the full set of assets
	if len(industry) == 1 and len(industry[0]) == 0:
		assets = industries.index.tolist()
	# Otherwise actually find subset
	else:
		flags = industries.apply(lambda x: x[0:len(industry[0])] == industry[0])
		for indx in industry:
			flags = flags | industries.apply(lambda x: x[0:len(indx)] == indx)
		assets = sorted(industries.index[flags].tolist())
	# Sort
	assets = sorted(assets)

	# Potentially exclude some sub-industries.
	# This is useful since we exclude FINREAL in some analyses.
	if to_exclude is not None:
		if isinstance(to_exclude, str):
			assets = [asset for asset in assets if to_exclude not in industries[asset]]
		elif isinstance(to_exclude, list):
			for x in to_exclude:
				assets = [asset for asset in assets if x not in industries[asset]]
		else:
			raise ValueError("to_exclude must be str or list")

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

	### get rid of redundant factors with all missing/zero exposures
	non_redundant = np.mean(
		np.isnan(exposures) | (exposures == 0), axis=0
	).mean(axis=0) != 1
	factor_cols = factor_cols[non_redundant]
	exposures = exposures[:, :, non_redundant]

	# Return output
	return dict(
		outcomes=outcomes[assets],
		exposures=exposures,
		industries=industries[assets],
		factor_cols=factor_cols,
	)

def main(args):
	# ensure cache directory exists
	os.makedirs(CACHE_DIR, exist_ok=True)
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
		df = pd.read_csv(
			f'{DATA_DIR}/bfre_factor_model_data_{ystart}_{yend}.csv',
			header=[0,1], index_col=0, nrows=nrows,
		)
		data.append(df)
	print(f"Finished loading all data at {elapsed(t0)}.")
	data = pd.concat(data, axis='index')
	all_assets = np.sort(np.unique([x[0] for x in data.columns if x[0][0] == 'Z']))

	# Load map from assets <--> names and vice versa
	asset_names = pd.read_csv(f"{DATA_DIR}/assets_id_to_name.csv").rename(
		columns={"invariant_id":"ASSET", "name_sec":"name"}
	)
	asset2names = asset_names.set_index("ASSET")['name']
	names2asset = asset_names.set_index("name")['ASSET']

	## 2. find factor columns
	factor_cols = np.sort([c for c in data[all_assets[0]].columns if c not in ['EXRETURN', 'SRET', 'CAPT']])
	k = len(factor_cols)
	if nrows is None:
		np.save(f"{CACHE_DIR}/factor_cols.npy", factor_cols)

	## 3. Create exposures
	print(f"Creating exposures at {elapsed(t0)}.")
	exposures = []
	for factor_col in factor_cols:
		exp = data[[(asset, factor_col) for asset in all_assets]].values
		exposures.append(exp)
	exposures = np.stack(exposures, axis=-1)

	## 4. Industry markers
	if nrows is None:
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

	## 5. Create returns (outcomes) (these are excess returns)
	print(f"Creating returns at {elapsed(t0)}.")
	# date parsing
	data.index = data.index.map(lambda x: datetime.datetime(
			year=int(str(x)[0:4]), month=int(str(x)[4:6]), day=int(str(x)[6:])
	))
	returns = data[[(asset, "EXRETURN") for asset in all_assets]].copy()
	returns.columns = returns.columns.get_level_values(0)
	returns.columns.name = 'ASSET'
	returns.index.name = 'Date'
	
	## 6. remove duplicate assets
	to_remove, _ = remove_redundant_assets(
		assets=all_assets, asset2names=asset2names, outcomes=returns.values, industries=industries
	)
	remaining_inds = [j for j, asset in enumerate(all_assets) if asset not in to_remove]
	exposures = exposures[:, remaining_inds]
	returns = returns.iloc[:, remaining_inds].copy()
	industries = industries.iloc[remaining_inds].copy()
	all_assets = all_assets[remaining_inds]
	if np.any(all_assets != returns.columns):
		raise RunTimeError("Alignment of return/exposure columns failed; there is a bug!")

	## 7. save everything
	if nrows is None:
		returns.to_csv(f"{CACHE_DIR}/returns.csv")
		industries.to_csv(f"{CACHE_DIR}/industries.csv")
		np.save(f"{CACHE_DIR}/exposures.npy", exposures)

	## 8. Exposure subset for simulations
	if nrows is None:
		target = datetime.datetime(year=2020, month=5, day=21)
	else:
		# Only used for debugging
		target = datetime.datetime(year=2018, month=1, day=2)
	ind = np.where(returns.index == target)[0].item()
	for industry in ['FIN', 'EGY']:
		sim_exposures = exposures[ind].copy()
		# subset to industry-specific assets
		sector_inds = np.array([
			i for i, asset in enumerate(all_assets) 
			if (industries[asset][0:3] == industry)
		])
		sim_exposures = sim_exposures[sector_inds].copy()
		# drop factors which are all zero or missing
		# (e.g. factors for non-financial sectors should be dropped when industry='FIN')
		sim_exposures = sim_exposures[:, ~np.all(np.isnan(sim_exposures) | (sim_exposures == 0), axis=0)]
		# ONLY for simulations, impute any missing exposures
		np.random.seed(123)
		sim_exposures[np.isnan(sim_exposures)] = np.random.choice(
			sim_exposures[~np.isnan(sim_exposures)].flatten(),
			np.sum(np.isnan(sim_exposures)),
			replace=True
		)
		if nrows is None:
			np.save(f"{CACHE_DIR}/simulation_exposures_{industry}.npy", sim_exposures)

if __name__ == '__main__':
	main(sys.argv)
