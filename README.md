# README

This repository contains the code necessary to replicate the paper "The Mosaic permutation test: an exact and nonparametric goodness-of-fit test for factor models."

## Dependency: ``mosaicperm``

This repository requires installation of the python package ``mosaicperm``---see https://mosaicperm.readthedocs.io/en/latest/.

## Real analysis of the BFRE model

All code used to analyze the model is available in the folder bfre_analysis/. In particular, ``bfre_preprocessing.py`` does the initial preprocessing and then ``bfre_analysis.ipynb`` does the main analysis and produces all of the plots in the paper.


The data for the analysis is not publicly available. However, we have made a synthetic sample dataset available in "data/bfre_placeholder/". To run the analysis on the sample synthetic dataset, do the following:
1. Download the synthetic exposures from https://drive.google.com/file/d/1583a9-U65h9MMovmh2I1WqTO0vMQTInR/view?usp=sharing and place the resulting file, called ``exposures.npy``, in the "data/bfre_placeholder/" folder.
2. Download the synthetic returns from https://drive.google.com/file/d/1mMejYeHBca4xARdR3127Etkv7Z_775qP/view?usp=sharing and place the resulting file, called ``returns.csv``, in the "data/bfre_placeholder/" folder.
3. Set the global variable "USE_PLACEHOLDER_DATA = True" in the first cell in ``bfre_analysis.ipynb``. 
4. After these two steps, you should be able to run the analysis in the notebook as-is using the synthetic data.

## Simulations

The simulations can be replicated as follows:
- Figure 3: see ``final_plots.ipynb``
- Figure 10: run ``sims/mpt_sims.sh``. The final plot is generated in ``final_plots.ipynb``