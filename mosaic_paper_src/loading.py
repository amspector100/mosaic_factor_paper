import numpy as np
import pandas as pd
from .src_context import root_dir

SIMULATION_DATA_PATH = root_dir / "data" / "bfre_cache"
PLACEHOLDER_DATA_PATH = root_dir / "data" / "bfre_placeholder"

def load_exposures(industry='FIN'):
	"""
	Loads the simulation exposures for a given industry.
	"""
	try:
		return np.load(SIMULATION_DATA_PATH / f"simulation_exposures_{industry}.npy")
	except FileNotFoundError:
		return np.load(PLACEHOLDER_DATA_PATH / f"simulation_exposures_{industry}.npy")

def load_garch_params(industry='FIN'):
	"""
	Loads the GARCH parameters for a given industry.
	"""
	try:
		return pd.read_csv(SIMULATION_DATA_PATH / f"garch_parameters_{industry}.csv")	
	except FileNotFoundError:
		return pd.read_csv(PLACEHOLDER_DATA_PATH / f"garch_parameters_{industry}.csv")

def load_mvn_arch_params(industry='FIN'):
	"""
	Loads the multivariate ARCH parameters for a given industry.
	"""
	try:
		return pd.read_csv(SIMULATION_DATA_PATH / f"multivariate_parameters_{industry}.csv")	
	except FileNotFoundError:
		return pd.read_csv(PLACEHOLDER_DATA_PATH / f"multivariate_parameters_{industry}.csv")