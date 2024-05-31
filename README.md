# Learning freely-described convex sets for Sylvester matrices
This is the MATLAB implementation for learning convex sets for Sylvester matrices based on the algorithm from [paper](https://arxiv.org/pdf/2307.04230).

## Prerequisites
Please install these packages and add them to MATLAB's path.
1. [YALMIP](https://yalmip.github.io/download/) with an SDP solver like [MOSEK](https://www.mosek.com/downloads/)
2. [Package to find null space of a sparse matrix](https://www.mathworks.com/matlabcentral/fileexchange/11120-null-space-of-a-sparse-matrix)
3. [sylvester](https://www.mathworks.com/matlabcentral/fileexchange/24124-sylvester-matrix): Package to embed polynomials into a square matrix
4. Create a `runs/` folder
5. Inside of the `runs/` folder, create an `experiments.xlsx` Excel spreadsheet. Also, create a worksheet with the same name as in the `cfg` struct in `run_syl_learn.m`. Results will get logged to the spreadsheet and variables/plots saved to the folder in `fpath`.

## Main files
1. `run_syl_learn.m`: hyperparameter and configuration file for running experiments
2. `syl_learn.m`: calls `syl_learn_alt_min.m` and evaluates the learned regularizer
3. `syl_learn_alt_min.m`: implements alternating minimization for learning regularizers for Sylvester matrices
4. `syl_learn_get_bases.m`: implements logic for finding bases for the linear maps
5. `syl_gen_dataset.m`: generates a dataset of Sylvester matrices

## Helper files
- `syl_evaluate_norm.m`: Evaluates regularizer on a matrix
- `syl_optimize_wrapper.m`: Wrapper for calling optimize
- `syl_recover.m`: Recovers with the regularizer
- `syl_recover_with_nuclear_norm.m`: Recovers with the nuclear norm regularizer

## Plotting files
- `syl_evaluation_histogram.m`: plots the distribution of regularizer evaluations
- `syl_box_plots.m`: plots the increase in minimum singular value as noise increases