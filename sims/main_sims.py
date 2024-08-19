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
from mosaic_paper_src import parser, utilities

# Specifies the type of simulation
DIR_TYPE = os.path.split(os.path.abspath(__file__))[1].split(".py")[0]

COLUMNS = [
	'seed',
	'n',
	'p',
	'k',
	'sparsity',
	'rho',
	'eps_dist',
	'test_stat',
	'test_stat_index',
	'method',
	'pval',
	'T',
	'runtime',
]
L_FILEPATH = "../data/bfre_cache/simulation_exposures.npy"
L_PLACEHOLDER_FILEPATH = "../data/bfre_placeholder/simulation_exposures.npy"

def sample_data(n, seed, rho, sparsity, L, eps_dist):
	p, k = L.shape
	np.random.seed(seed)
	# scipy dist used to draw samples
	if eps_dist == 'laplace':
		eps_dist = stats.laplace()
	elif eps_dist == 'tdist':
		eps_dist = stats.t(df=4)
	elif eps_dist == 'gaussian':
		eps_dist = stats.norm()
	else:
		raise ValueError(f"Unrecognized eps_dist={eps_dist}.")
	# Factor returns
	X = eps_dist.rvs(size=(n,k))
	# Alternative
	s0 = max(2, int(np.ceil(sparsity * p)))
	non_nulls = np.random.choice(np.arange(p), s0, replace=False)
	v = np.zeros(p); v[non_nulls] = 1 / np.sqrt(len(non_nulls))
	# Returns
	gamma = eps_dist.rvs(size=(n,p))
	Z = eps_dist.rvs(size=n)
	eps = gamma + rho * Z.reshape(-1, 1) * v.reshape(1, -1)
	Y = X @ L.T + eps
	return dict(
		Y=Y,
		eps=eps,
		gamma=gamma,
		v=v,
		rho=rho,
	)

def append_mpt_results(
	output,
	factor_test,
	method,
	test_stat_name,
	dgp_args,
	runtime
):
	# Append final p-value
	output.append(
		dgp_args + [
			test_stat_name,
			'adaptive',
			method,
			factor_test.pval,
			factor_test.adapt_stat,
			runtime
		]
	)
	if len(factor_test.statistic) > 1:
		stat = factor_test.statistic
		null_stats = factor_test.null_statistics
		nrand = len(null_stats)
		for i in range(len(stat)):
			pvali = (1 + np.sum(stat[i] <= null_stats[:, i])) / (nrand + 1) 
			output.append(
				dgp_args + [
					test_stat_name,
					i,
					method,
					pvali,
					stat[i],
					runtime
				]
			)
	return output

def split_mse_stats_v2(
	hateps,
	partitions,
	vs,
):
	# Preprocessing
	mus = hateps.mean(axis=0)
	vs_adj = [v[:, 0] for v in vs]
	n, p = hateps.shape
	
	# initialize output
	baseline_error = np.sum(hateps**2)
	oos_errors = np.zeros(len(vs))
	oos_resids = np.zeros(hateps.shape)
	for i, v in enumerate(vs_adj):
		# special case
		if np.all(v == 0):
			oos_errors[i] = baseline_error
			continue
		# compute out-of-sample resids
		for nstart, nend, subset in partitions:
			# Predict Z
			negsub = np.ones(p, dtype=bool); negsub[subset] = False
			hatZ = hateps[nstart:nend, negsub] @ v[negsub] + mus[subset] @ v[subset]
			hatZ /= np.sum(v**2)
			# Predict hateps[:, sub]
			preds = hatZ.reshape(-1, 1) * v[subset].reshape(1, -1)
			oos_resids[nstart:nend, subset] = hateps[nstart:nend, subset] - preds
		
		# convolve
		oos_errors[i] = np.sum(oos_resids**2)
	
	# Compute r2s in sliding window
	return 1 - oos_errors / baseline_error

def single_seed_sim(
	seed, n, L, sparsity, rho, eps_dist, t0, **args
):
	# arguments and defaults
	p, k = L.shape
	dgp_args = [
		seed, n, p, k, sparsity, rho, eps_dist
	]
	# method arguments
	msg = f"At seed={seed}, n={n}, sparsity={sparsity}, rho={rho}"
	msg += f" at {utilities.elapsed(t0)}."
	print(msg)
	sys.stdout.flush()

	# create data
	data = sample_data(n=n, seed=seed, rho=rho, sparsity=sparsity, L=L, eps_dist=eps_dist)

	# initialize output
	output = []

	## Test stat = mmc_stat
	nrand = args.get("nrand", 200)
	# default FRT + OLS with no reps;
	# we do postprocessing later to compute 
	# the real OLS oracle p-value
	for test_stat, test_stat_name in zip(
		[mp.statistics.quantile_maxcorr_stat],
		['quant_corr'],
	):
		for ngroups, method, nreps in zip(
			[None, 1],
			['MPT', 'OLS oracle'],
			[nrand, 1]
		):
			time0 = time.time()
			mptest = mp.factor.MosaicFactorTest(
				outcomes=data['Y'],
				exposures=L,
				test_stat=test_stat,
				ngroups=ngroups,
			)
			mptest.fit(nrand=nreps, verbose=False)
			mpt_runtime = time.time() - time0
			append_mpt_results(
				output,
				factor_test=mptest,
				method=method,
				test_stat_name=test_stat_name,
				dgp_args=dgp_args,
				runtime=mpt_runtime, 
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

	## Load exposures
	if os.path.exists(L_FILEPATH):
		L = np.load(L_FILEPATH)
		print(L.shape)
	else:
		print("Simulation exposures are not available---using placeholder instead.")
		L = np.load(L_PLACEHOLDER_FILEPATH)

	## Key defaults go here
	args['n'] = args.get("n", [100])
	args['rho'] = args.get("rho", [0])
	args['sparsity'] = args.get("sparsity", [0.0])
	args['eps_dist'] = args.get("eps_dist", ['laplace'])

	# Save args, create output dir
	output_dir = utilities.create_output_directory(args, dir_type=DIR_TYPE)
	args.pop("description")

	# Run outputs
	outputs = utilities.apply_pool_factorial(
		func=single_seed_sim,
		seed=list(range(seed_start, reps+seed_start)), 
		num_processes=num_processes,
		t0=[t0],
		L=[L],
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
		'p',
		'k',
		'sparsity',
		'rho',
		'method',
		'test_stat',
		'test_stat_index',
	])[['disc', 'pval', 'T', 'runtime']].agg(['mean'])
	pd.set_option('display.max_rows', 500)
	print(summary)


if __name__ == '__main__':
	main(sys.argv)