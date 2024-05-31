%% Sets up hyperparameters and configuration for running experiments

clear all, close all, clc

% By default, don't show plots
set(0, 'DefaultFigureVisible', 'off');

% Set tolerance
delta = 1e-10;

% Define hyperparameters
hp = struct( ...
    ... % Options: EP1, EPNN, MG, SRV, SRV_NMR, SRV_NAR
    ... % EP1: function fitting to 1
    ... % EPNN: function fitting to nuclear norm values
    ... % MG: minimize primal-dual gap on recovery problems
    ... % SRV: separate regularizer values, constrain "good" points while maximizing "bad" points, randomly drawn from matrices of squared norm n
    ... % SRV_NMR: like SRV, except each matrix is normalized by the maximum row norm
    ... % SRV_NAR: like SRV, except each row of each matrix is normalized to have norm 1
    "approach", "EP1", ...
    ... % Load variables from a previous run stored in a .mat file under runs/
    "from_pretrained", "", ...
    "random_seed", 3, ...       % Random seed for reproducibility
    "impose_ext", 1, ...        % Impose extendability conditions (set to true to extend to higher dimensions)
    "N", 1000, ...              % Number of training matrices
    "n", 2, ...                 % Number of polynomials per training matrix
    "max_n", 5, ...             % Maximum number of polynomials to extend to
    "k", 3, ...                 % Degree of the polynomials
    "d", 1, ...                 % Degree of the common factor polynomial
    "noise", 1e-2, ...          % std of noise on each entry of matrix
    "RW", 1, ...                % regularization weight during training (MG approach)
    "lambda_min", 1e-4, ...     % min value for lambda
    "lambda_init", 10, ...      % initial lambda size
    "a_init_std", 1e-3, ...     % initial a std
    "b_init_std", 1e-3, ...     % initial b std
    "alpha", 0.5, ...           % damping: amount to stay on old A, B, and lambda
    "num_alts", 75, ...         % max number of alternations
    "num_inits", 1, ...         % number of initializations  % kept as legacy
    "B_bnd", 1e4, ...               % norm bound on B, needed for stability during alternation
    "delta", delta, ...             % precision on MOSEK solver
    "batch_size", 100, ...          % 'all' or number of points per batch
    ... % Description space size: W_n
    "largeB", false, ...        % If true, use W_n=U_n (worse than W_n=V_n empirically)
    "largeB_iter", 1, ...       % Number of iterations on W_n=V_n before increasing to W_n=U_n
    "largeB_jolt", 3e-3, ...    % Add noise to warm-start
    ... % Description space size: Increasing number of copies (another Ax+By+tu>=0 constraint)
    "num_starting_copies", 1, ...        % number of copies to start on, recommend 1
    "tot_copies", 1, ...                 % total copies, number of copies increases 1 at a time until either tot_copies or num_alts is reached  
    "num_iter_per_extra_copy", 200, ...  % number of iterations to wait until another copy is added
    "copy_init_magnitude_factor", 5, ... % initialize next copy with std that is "copy_init_magnitude_factor" less
    ... % Termination criteria
    "err_rel_tol", 1e-8, ...        % threshold on relative change in error
    "err_consec_iter_bnd", 50, ...  % patience, number of consecutive iterations before terminating
    "err_abs_thresh", 1e-4, ...     % stops when absolute error is less than this
    ... % Adam optimization method - not experimented thoroughly
    "use_adam", false, ...      % flag to employ Adam
    "beta1", 0.99, ...          % coefficient for first-order moving average of gradient
    "beta2", 0.99 ...           % coefficient for second-order moving average of gradient
    );


% Define config
cfg = struct( ...
    'fpath', ['runs/run_'], ... % path to folders containing experiments
    'row', 2, ...               % row number for recording experiment on Excel spreadsheet
    'test_size', 100, ...       % Number of test data points
    'spreadsheet', 'runs/experiments.xlsx', ... % Spreadsheet to write to
    'worksheet', 'EP1', ...                     % Worksheet to write to
    'plot_comparisons', false, ...              % Whether to plot comparisons with nuclear norm regularizer
    'eval', true ...                            % Whether to evaluate on the test set
    );


%% Run Experiments
ops = sdpsettings('solver','mosek','verbose', 0, ...
                  'mosek.MSK_DPAR_INTPNT_CO_TOL_DFEAS', hp.delta, ...
                  'mosek.MSK_DPAR_INTPNT_CO_TOL_PFEAS', hp.delta, ...
                  'mosek.MSK_DPAR_INTPNT_CO_TOL_MU_RED', hp.delta, ...
                  'mosek.MSK_DPAR_INTPNT_TOL_DFEAS', hp.delta, ...
                  'mosek.MSK_DPAR_INTPNT_CO_TOL_REL_GAP', hp.delta);


fprintf("Training\n");
cfg.row = 2; cfg.fpath = get_new_folder(); cfg.row = verify_cfg_row(cfg);
rng(hp.random_seed);
syl_learn(hp, cfg, ops);




function [path] = get_new_folder()
    % Get new folder
    current_dt = datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss');
    formatted_dt = datestr(current_dt, 'yyyy-mm-dd_HH-MM-ss');
    fpath = ['runs/run_' formatted_dt '/'];
    fprintf("Saving variables to %s\n", fpath);
    path = ['runs/run_' formatted_dt '/'];
end


function [row] = verify_cfg_row(cfg)
    % Changes the row in cfg to the first blank space in spreadsheet
    [~, ~, raw] = xlsread(cfg.spreadsheet, cfg.worksheet);
    raw_size = size(raw);
    num_rows = raw_size(1);
    if (num_rows >= cfg.row)
        fprintf("Setting would overwrite data, changing row to %d\n", num_rows + 1);
        row = num_rows + 1;
    else
        row = cfg.row;
    end

end