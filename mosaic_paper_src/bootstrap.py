import numpy as np
from typing import Optional
from .src_context import mosaicperm as mp

METHODS = ['default', 'block', 'within_block']

def _create_resampling_inds(
        T: int,
        sample_size: int,
        block_size: Optional[int]=None,
        method: str='default',
):
    """
    Create indices for resampling residuals.
    """
    if method == 'default':
        return np.random.randint(0, T, size=sample_size)
    elif method == 'block':
        assert T % block_size == 0, "Block size must divide the number of timepoints"
        assert sample_size % block_size == 0, "Block size must divide the total size"
        n_blocks = T // block_size
        # List of blocks
        blocks = np.random.randint(0, n_blocks, size=sample_size // block_size)
        # make block_size times longer, so if block_size=2, then [1,2,3] becomes [1,1,2,2,3,3]
        output = (blocks.reshape(-1, 1) * np.ones((1, block_size))).flatten(order='C')
        # remainder indices
        remainder = np.arange(sample_size) % block_size
        return (block_size * output + remainder).astype(int)
    elif method == 'within_block':
        assert T % block_size == 0, "Block size must divide the number of timepoints"
        assert sample_size % T == 0, "T must divide the total size"
        # Block numbers: so if block_size=2, this looks like [0, 0, 2, 2, 4, 4, ...] 
        blocks = block_size * (np.arange(sample_size) // block_size)
        # Choose random indices within each block
        # so if block_size=2, this could look like [0, 1, 3, 3, 5, 4, ...]
        inds = blocks + np.random.randint(0, block_size, size=sample_size)
        # Ensure all < T; this is important when sample_size > T
        inds = inds % T
        return inds
    else:
        raise ValueError(f"Invalid method: {method}")

def _resample_residuals(
    hateps: np.array,
    block_size: Optional[int]=None,
    impose_null: bool=True,
    method: str='default',
):
    """
    Resample residuals from the model.
    This is actually a bottleneck, so we use fancy indexing for speed.
    """
    T, p = hateps.shape
    # Under the null, resample columns independently
    if impose_null:
        hateps_bs = hateps[
            (_create_resampling_inds(T, T*p, block_size=block_size, method=method),
                np.arange(T*p) // T)
        ].reshape(T, p, order='F') # order='F' since the columns are contiguous here
    else:
        # without imposing the null, resample each row of the residuals 
        hateps_bs = hateps[_create_resampling_inds(T, sample_size=T, block_size=block_size)]
    return hateps_bs


def _bootstrap_residuals(
    hateps: np.array,
    impose_null: bool=True,
    residualize: bool=True,
    method: str='default',
    exposures: Optional[np.array]=None,
    block_size: Optional[int]=None,
):
    T, p = hateps.shape
    # Under the null, resample columns independently    
    hateps_bs = _resample_residuals(hateps, block_size=block_size, impose_null=impose_null, method=method)
    # Possibly residualize
    if residualize:
        if exposures is None:
            raise ValueError("Exposures must be provided if residualizing")
        if len(exposures.shape) == 3 or impose_null:
            hateps_bs = mp.factor.ols_residuals(hateps_bs, exposures)
        else:
            pass # hateps_bs is already residualized when exposures are 2D and impose_null=False
    return hateps_bs

def bootstrap_test_stat(
   outcomes: np.array,
   test_stat: callable,
   exposures: np.array,
   hateps: Optional[np.array]=None, 
   n_bootstraps: int=100,
   # bootstrap configuration
   block_size: Optional[int]=None,
   impose_null: bool=True,
   residualize: bool=True,
   method: str='default',
):
    # Create residuals
    if hateps is None:
        hateps = mp.factor.ols_residuals(outcomes, exposures)
    # Create test statistic
    statistic = test_stat(hateps)
    # Bootstrap
    bootstrap_stats = np.zeros(n_bootstraps)
    for i in range(n_bootstraps):
        hateps_bs = _bootstrap_residuals(
            hateps,
            impose_null=impose_null,
            residualize=residualize,
            exposures=exposures,
            block_size=block_size,
            method=method,
        )
        bootstrap_stats[i] = test_stat(hateps_bs)
    return statistic, bootstrap_stats


def naive_permutation_test(hateps, test_stat, R=100):
    """
    Perform naive permutation test on hateps OLS.
    """
    n, p = hateps.shape
    S = test_stat(hateps)
    hatepsr = hateps.copy()
    inds = np.arange(n)
    S0s = np.zeros(R)
    for r in range(R):
        # Create indices to shuffle each column independently
        xinds = np.concatenate([np.random.permutation(inds) for _ in range(p)])
        yinds = np.arange(n*p) // n
        # Shuffle
        hatepsr = hateps[(xinds, yinds)].reshape(n, p, order='F')
        S0s[r] = test_stat(hatepsr)
    pval = (1 + np.sum(S <= S0s)) / (R + 1)
    return pval, S, S0s