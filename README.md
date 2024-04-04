# README

This repository contains the code necessary to replicate the paper "The Mosaic permutation test: an exact and nonparametric goodness-of-fit test for factor models".

## Dependency: ``mosaicperm``

This repository requires installation of the python package ``mosaicperm``---see https://mosaicperm.readthedocs.io/en/latest/.

## Real analysis of the BFRE model

The data for the BFRE model is not publicly available, but all code used to analyze the model is available in the folder bfre_analysis/. In particular, ``bfre_preprocessing.py`` does the initial preprocessing and then ``bfre_analysis.ipynb`` does the main analysis and produces all of the plots in the paper.

## Simulations

The simulations can be replicated as follows:
- Figure 3: see ``final_plots.ipynb``
- Figure 10: run ``sims/mpt_sims.sh``