[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18774474.svg)](https://doi.org/10.5281/zenodo.18774474) 

# SNNLS for AC Susceptibility-Based Hydrodynamic Size Estimation

Spline-Regularized Non-Negative Least Squares (SNNLS) implementation for hydrodynamic size distribution reconstruction from AC susceptibility data.
This repository contains the MATLAB implementation of the Spline-Regularized Non-Negative Least Squares (SNNLS) framework described in:

Mohd Mawardi Saari et al.,
"Estimation of hydrodynamic size distribution from AC susceptibility: A model-free approach using spline-regularized non-negative least squares",
Measurement, 2026.

## Repository Contents
- Main File to run the SNNLS function
- Synthetic data generator function for AC susceptibility simulation
- Implementation of:
  - SNNLS (Spline-Regularized NNLS)
  - Classical histogram-based NNLS
  - Tikhonov-regularized NNLS
  - Lognormal distribution fitting
- Comparison and visualization scripts

The provided synthetic data generator enables full reproduction of the algorithmic validation and method comparison presented in the manuscript.

## How to Run

Run:
MainProgramFile.m

## License

MIT License
