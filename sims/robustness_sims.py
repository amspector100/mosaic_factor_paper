"""
Runs simulations on the factor randomization test.
"""

import os
import sys
import time

import numpy as np
from scipy import stats
import pandas as pd
from context import mosaicperm as mp
from context import mosaic_paper_src
from mosaic_paper_src import parser, utilities, nonexch_sampling
from bootstrap_sims import load_exposures

# Specifies the type of simulation
DIR_TYPE = os.path.split(os.path.abspath(__file__))[1].split(".py")[0]

COLUMNS = [
	'seed',
	'n',
	'industry',
	'sampling_method',
	'method',
	'statistic',
	'pval',
	'zstat',
	'null_stat',
]
SIMULATION_DATA_PATH = "../data/bfre_cache/"
PLACEHOLDER_DATAH_PATH = "../data/bfre_placeholder/"

def load_garch_params(industry='FIN'):
	"""
	Loads the GARCH parameters for a given industry.
	"""
	try:
		return pd.read_csv(SIMULATION_DATA_PATH + f"garch_parameters_{industry}.csv")	
	except FileNotFoundError:
		return pd.read_csv(PLACEHOLDER_DATAH_PATH + f"garch_parameters_{industry}.csv")

def load_mvn_arch_params(industry='FIN'):
	"""
	Loads the multivariate ARCH parameters for a given industry.
	"""
	try:
		return pd.read_csv(SIMULATION_DATA_PATH + f"multivariate_parameters_{industry}.csv")	
	except FileNotFoundError:
		return pd.read_csv(PLACEHOLDER_DATAH_PATH + f"multivariate_parameters_{industry}.csv")

def single_seed_sim(
	seed, n, industry, sampling_method, t0, **args
):
	industry = industry.upper()
	# # arguments and defaults
	dgp_args = [
		seed, n, industry, sampling_method,
	]
	# # method arguments
	msg = f"At seed={seed}, n={n}"
	msg += f" at {utilities.elapsed(t0)}."
	print(msg)
	sys.stdout.flush()

	# data (placeholder for now)
	np.random.seed(seed)
	exposures = load_exposures(industry=industry) # p x k
	garch_params = load_garch_params(industry=industry)
	mvn_arch_params = load_mvn_arch_params(industry=industry)

    # simulate residuals
	if sampling_method == 'ar1':
		outcomes = nonexch_sampling.simulate_ar1_residuals(T=n, n=exposures.shape[0], phi=garch_params['rho'].values).T
	elif sampling_method == 'garch':
		outcomes = nonexch_sampling.simulate_garch_residuals(
			T=n,
			n=exposures.shape[0],
			omegas=garch_params['omega'].values,
			alphas=garch_params[['alpha1', 'alpha2', 'alpha3']].values,
			betas=garch_params[['beta1', 'beta2', 'beta3']].values,
			rho=np.zeros(exposures.shape[0]),
		).T
	elif sampling_method == 'garch_ar1':
		outcomes = nonexch_sampling.simulate_garch_residuals(
			T=n,
			n=exposures.shape[0],
			omegas=garch_params['omega'].values,
			alphas=garch_params[['alpha1', 'alpha2', 'alpha3']].values,
			betas=garch_params[['beta1', 'beta2', 'beta3']].values,
			rho=garch_params['rho'].values,
		).T
	elif sampling_method == 'mvn_arch':
		outcomes = nonexch_sampling.simulate_multivariate_diagonal_arch(
			T=n,
			omegas=mvn_arch_params.values[:, 0],
			A=mvn_arch_params.values[:, 1:],
			rho=garch_params['rho'].values,
		).T
	else:
		raise ValueError(f"Unrecognized sampling_method={sampling_method}.")

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
		max_batchsize=2,
	)
	mptest.fit(nrand=nrand, verbose=False)
	output.append(
		dgp_args + ['MPT', mptest.statistic, mptest.pval, mptest.apprx_zstat, mptest.null_statistics[0].item()]
	)

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
	args['rho'] = args.get("rho", [0.5])
	args['sampling_method'] = args.get("sampling_method", ['ar1'])

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
		'sampling_method',
		'industry',
	])[['pval', 'statistic', 'null_stat', 'zstat']].agg(['mean'])
	pd.set_option('display.max_rows', 500)
	print(summary)


if __name__ == '__main__':
	main(sys.argv)