import numpy as np

# def fit_factor_model_als(y, max_iter=100, tol=1e-6):
#     """
#     Fits a single-factor model using alternating least squares.
    
#     The model is:
#     Y_{i,t} = sigma_i * lambda_t + noise
    
#     where sigma_i are factor loadings and lambda_t are factor values.
    
#     Parameters:
#     -----------
#     y : np.ndarray
#         Input data of shape (n, p) where n is the number of entities and p is the number of time points
#     max_iter : int, optional
#         Maximum number of iterations (default is 100)
#     tol : float, optional
#         Convergence tolerance (default is 1e-6)
    
#     Returns:
#     --------
#     sigma : np.ndarray
#         Estimated factor loadings of shape (n,)
#     lambda_t : np.ndarray
#         Estimated factor values of shape (p,)
#     """
#     n, p = y.shape
    
#     # Initialize sigma and lambda
#     sigma = np.ones(n)
#     lambda_t = np.nanmean(y, axis=0)
    
#     prev_loss = np.inf
    
#     for iteration in range(max_iter):
#         # Update sigma given lambda
#         sigma = np.sum(y * lambda_t, axis=1) / np.sum(lambda_t**2)
        
#         # Update lambda given sigma
#         lambda_t = np.sum(y.T * sigma, axis=1) / np.sum(sigma**2)
        
#         # Calculate loss
#         y_hat = np.outer(sigma, lambda_t)
#         loss = np.mean((y - y_hat)**2)
        
#         # Check convergence
#         if np.abs(loss - prev_loss) < tol:
#             break
            
#         prev_loss = loss
    
#     return sigma, lambda_t

def simulate_garch_residuals(n, T, omegas, alphas, betas, rho):
    """
    Simulate independent residual trajectories from a GARCH(k,k) model with exponentially decaying parameters
    and AR(1) innovations.
    
    Parameters:
    n : int
        number of trajectories to simulate
    T : int
        length of the trajectory
    omegas : float
        n_trajectoris-array of constant term in the variance equation
    alphas : float
        (n, k) level for ARCH coefficients
    betas : float
        (n, k) level for GARCH coefficients
    rho : np.array
        (n,) array of AR(1) coefficients
            
    Returns:
    epsilon : np.array
        simulated series (innovations scaled by sqrt of conditional variance)
    """
    # Ensure alphas and betas are 2D arrays
    if len(alphas.shape) == 1:
        alphas = alphas.reshape(1, -1)
    if len(betas.shape) == 1:
        betas = betas.reshape(1, -1)
    if betas.shape[1] != alphas.shape[1]:
        raise ValueError("alphas and betas must have the same number of columns (only Garch(k,k) is supported).")
    k = betas.shape[1]

    # Initialize    
    epsilon = np.zeros((n, T+k))
    sigma2 = np.zeros((n, T+k))
    
    # Calculate unconditional variance to use as starting value.
    stationary_flags = (alphas.sum(axis=1) + betas.sum(axis=1)) < 1
    unc_var = np.zeros(n)
    unc_var[stationary_flags] = omegas[stationary_flags] / (1 - alphas[stationary_flags].sum(axis=1) - betas[stationary_flags].sum(axis=1))
    unc_var[~stationary_flags] = 1
    sigma2[:,:k] = unc_var.reshape(-1, 1)
    epsilon[:,:k] = np.sqrt(unc_var.reshape(-1, 1)) * np.random.normal(size=(n, k))
    
    # Generate standard normal innovations for t >= q
    z = np.random.randn(n, T+k)
    
    # Simulate the GARCH(k,k) process
    for t in range(k, T+k):
        # Calculate contribution from past k residuals and variances
        arch_term = np.sum(alphas * epsilon[:,t-k:t][:,::-1]**2, axis=1)
        garch_term = np.sum(betas  * sigma2[:,t-k:t][:,::-1], axis=1)
        sigma2[:,t] =  omegas + arch_term + garch_term
        epsilon[:,t] = np.sqrt(1-rho**2) * z[:,t] * np.sqrt(sigma2[:,t]) + rho * epsilon[:, t-1]
        
    return epsilon[:, k:]

def simulate_ar1_residuals(T, n, phi):
    """
    Simulate n trajectories of an AR(1) process in parallel.
    
    The process is defined as:
      x_t = phi * x_{t-1} + np.sqrt(1-phi**2) * epsilon_t,
    where epsilon_t ~ N(0,1) and the process starts from its stationary distribution.
    
    Parameters:
    - T: int, length of each time series
    - n: int, number of trajectories (samples) to simulate
    - phi: float or np.array, AR(1) coefficient (|phi| < 1 for stationarity)
    - sigma: float, standard deviation of the white noise
    
    Returns:
    - x: np.array of shape (n, T), where each row is a simulated trajectory
    """
    if isinstance(phi, float):
        if phi >= 1:
            raise ValueError(f"phi={phi}>=1 does not satisfy the stationarity condition.")
    if isinstance(phi, np.ndarray):
        if np.any(phi > 1):
            raise ValueError(f"phi={phi}>=1 does not satisfy the stationarity condition.")
    
    # Initialize the array to hold the trajectories.
    x = np.zeros((n, T))

    # Residual variance
    tau = np.sqrt(1-phi**2)
    
    # Initialize the first time step from the stationary distribution.
    x[:, 0] = np.random.normal(size=n)
    
    # Generate the trajectories in a vectorized loop.
    for t in range(1, T):
        x[:, t] = phi * x[:, t-1] + tau * np.random.normal(size=n)
    
    return x

def simulate_multivariate_diagonal_arch(T, omegas, A, rho):
    """
    Simulate a multivariate ARCH process with autocorrelation
    but no cross-sectional correlation.
    
    Parameters:
    - T: int, length of each time series
    - A: np.array, shape (n, n), ARCH coefficients
    """
    n = A.shape[0]
    epsilon = np.zeros((n, T))
    
    # Initialize the first time step from the stationary distribution.
    epsilon[:, 0] = np.random.normal(size=n)
    z = np.random.randn(n, T)
    
    # Generate the trajectories in a vectorized loop.
    for t in range(1, T):
        sigma2s = np.maximum(A @ epsilon[:, t-1]**2, 0)
        epsilon[:, t] = z[:, t] * np.sqrt(omegas + sigma2s)

    return epsilon
    
    
    