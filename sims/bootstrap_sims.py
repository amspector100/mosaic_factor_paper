"""
Runs simulations on the factor randomization test.
"""

import os
import sys
import time
import datetime
from pathlib import Path
import numpy as np
from scipy import stats
import pandas as pd
from context import mosaicperm as mp
from context import mosaic_paper_src, root_directory
from mosaic_paper_src import parser, utilities, bootstrap

# Specifies the type of simulation
DIR_TYPE = os.path.split(os.path.abspath(__file__))[1].split(".py")[0]

COLUMNS = [
	'seed',
	'industry',
	'n',
	'method',
	'statistic',
	'pval',
	'zstat',
	'null_stat',
]

SIMULATION_DATA_PATH = Path(root_directory) / "data" / "bfre_cache"
PLACEHOLDER_DATA_PATH = Path(root_directory) / "data" / "bfre_placeholder"

def load_exposures(industry='FIN'):
	"""
	Loads the simulation exposures for a given industry.
	"""
	try:
		return np.load(SIMULATION_DATA_PATH / f"simulation_exposures_{industry}.npy")	
	except FileNotFoundError:
		return np.load(PLACEHOLDER_DATA_PATH / f"simulation_exposures_{industry}.npy")

def load_sigma2s(industry='FIN'):
	"""
	Loads the simulation sigma2s for a given industry.
	"""
	try:
		df = pd.read_csv(SIMULATION_DATA_PATH / f"simulation_sigma2s_{industry}.csv", index_col=0)	
	except FileNotFoundError:
		df = pd.read_csv(PLACEHOLDER_DATA_PATH / f"simulation_sigma2s_{industry}.csv", index_col=0)
	df.index = pd.to_datetime(df.index)
	return df

def single_seed_sim(
	seed, n, industry, t0, **args
):
	industry = industry.upper()
	# # arguments and defaults
	dgp_args = [
		seed, industry, n, 
	]
	# # method arguments
	msg = f"At seed={seed}, n={n}"
	msg += f" at {utilities.elapsed(t0)}."
	print(msg)

	# data (placeholder for now)
	np.random.seed(seed)
	exposures = load_exposures(industry=industry)
	sigma2s = load_sigma2s(industry=industry)
	# Find n days which are closest to covid	
	covid = datetime.datetime(2020, 2, 20)
	distances = np.abs((sigma2s.index - covid).days)
	order = np.argsort(distances.astype(float))
	selected_days = sigma2s.index[order[:n]].sort_values()
	# Create sigma2s
	sigma2s = sigma2s.loc[selected_days].values
	# use weeks as batches
	isocal = pd.Series(selected_days.sort_values()).dt.isocalendar()
	weeks = isocal['week'] + 52 * isocal['year']
	batches = [np.where(weeks == i)[0] for i in np.unique(weeks)]
	# Create outcomes
	outcomes = np.random.randn(n, exposures.shape[0]) * np.sqrt(sigma2s)
	print(f"Finished loading at {utilities.elapsed(t0)}.")

	# initialize output
	output = []

	# Args
	nrand = args.get("nrand", 200)
	test_stat = mp.statistics.mean_maxcorr_stat

	# Run mosaic permutation test
	mptest = mp.factor.MosaicFactorTest(
		outcomes=outcomes,
		exposures=exposures,
		test_stat=test_stat,
		batches=batches,
	)
	mptest.fit(nrand=nrand, verbose=False)
	output.append(
		dgp_args + ['MPT', mptest.statistic, mptest.pval, mptest.apprx_zstat, mptest.null_statistics[0].item()]
	)
	print(f"Finished MPT at {utilities.elapsed(t0)}.")

	# Naive permutation test
	pval, statistic, null_stats = bootstrap.naive_permutation_test(
		hateps=mp.factor.ols_residuals(outcomes, exposures),
		test_stat=test_stat,
		R=nrand,
	)
	output.append(
		dgp_args + [
			'Naive Permutation',
			statistic,
			pval,
			(statistic - null_stats.mean()) / null_stats.std(),
			null_stats[0].item(),
		]
	)

	# Run bootstraps
	impose_null = True
	for residualize in [True]:
		for method in ['default', 'within_block', 'block']:
			statistic, bootstrap_stats = bootstrap.bootstrap_test_stat(
				outcomes=outcomes,
				test_stat=test_stat,
				exposures=exposures,
				impose_null=impose_null,
				block_size=args.get("block_size", 5),
				residualize=residualize,
				n_bootstraps=nrand,
				method=method,
			)
			for use_quantile in [True]:
				if use_quantile:
					pval = (1 + np.sum(statistic <= bootstrap_stats)) / (nrand + 1)
				else:
					pval = 1 - stats.norm.cdf(
						(statistic - bootstrap_stats.mean()) / bootstrap_stats.std()
					)
				method = 'bootstrap' + '_residualize' + str(residualize) + '_method' + str(method)
				method += '_quantile' + str(use_quantile)
				output.append(dgp_args + [
					method,
					statistic,
					pval,
					(statistic - bootstrap_stats.mean()) / bootstrap_stats.std(),
					bootstrap_stats[0].item(),
				])
			print(f"Finished {method} at {utilities.elapsed(t0)}.")

	return output

def main(args):
	t0 = time.time()
	# Parse arguments
	args = parser.parse_args(args)
	reps = args.pop('reps', [1])[0]
	seed_start = args.pop('seed_start', [1])[0]
	num_processes = args.pop('num_processes', [1])[0]
	# parse job id
	job_id = int(args.pop("job_id", [0])[0])

	## Key defaults go here
	args['n'] = args.get("n", [100])
	args['industry'] = args.get("industry", ['FIN'])

	# Save args, create output dir
	output_dir = utilities.create_output_directory(args, dir_type=DIR_TYPE)
	args.pop("description")

	# Run outputs
	outputs = utilities.apply_pool_factorial(
		func=single_seed_sim,
		seed=list(range(seed_start, reps+seed_start)), 
		num_processes=num_processes,
		t0=[t0],
		**args,
	)
	# concatenate to df
	out_df = []
	for x in outputs:
		out_df.extend(x)
	out_df = pd.DataFrame(out_df, columns=COLUMNS)
	out_df.to_csv(output_dir + f"results_id{job_id}_seedstart{seed_start}.csv", index=False)

	# print
	out_df['disc'] = out_df['pval'] <= 0.1
	summary = out_df.groupby([
		'n',
		'method',
		'industry',
	])[['pval', 'statistic', 'null_stat', 'zstat']].agg(['mean'])
	pd.set_option('display.max_rows', 500)
	print(summary)


if __name__ == '__main__':
	main(sys.argv)