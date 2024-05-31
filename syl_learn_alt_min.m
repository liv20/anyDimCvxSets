function [res] = syl_learn_alt_min(Xtrues_train, Xtils_train, fvals_train, hp, bases, ops)
% Function: syl_learn_alt_min
%
% Description: runs alternating minimization
%
% Note: Only supports loading from hp.pre_trained without largeB
%
% Input:
%   - Xtrues_train: training matrices with the approximate GCD structure
%   - Xtils_train: training matrices with perturbed values. If hp.approach
%       is in the SRV class, these matrices come from the negative
%       distribution.
%   - fvals_train: the value the regularizer should evaluate Xtrues_train
%       to
%   - hp: hyperparameters, see run_syl_learn.m for options
%   - bases: bases for the learned values in a and b variables that convert
%       them into the linear maps A and B
%   - ops: optimization options
% Output:
%   - res: result of the alternating minimization

vec = @(x) x(:);

% Set up addFittedPointConstraint variables
err_vars = sdpvar(1,hp.batch_size,'full');

% Set up gauge function variables
t_vars = sdpvar(hp.tot_copies,hp.N,'full');
y_vars = sdpvar(hp.n*(hp.k+1), hp.N, 'full');
z_vars = sdpvar(max(bases.N_U),max(bases.N_U),hp.tot_copies,hp.N,'symmetric');

% Set up parameter variables
a_vars = sdpvar(size(bases.A,2),hp.tot_copies,'full');
b_vars = sdpvar(size(bases.B,2),hp.tot_copies,'full');
A_vars = reshape(bases.A*a_vars,max(bases.N_U)^2,[], hp.tot_copies);
B_vars = reshape(bases.B*b_vars,max(bases.N_U)^2,[], hp.tot_copies);
lambda_var = sdpvar(1,1);  % regularization parameter

% Set up number of copies
hp.num_copies = hp.num_starting_copies;

% Initialize parameter values
if hp.from_pretrained == ""
    % Randomly initialize A, B, lambda
    aval = randn(size(bases.A,2),hp.num_starting_copies) * hp.a_init_std;
    bval = randn(size(bases.B,2),hp.num_starting_copies) * hp.b_init_std;
    lambda = hp.lambda_init;
else
    % Load A, B, lambda from pre-trained parameters
    res = load(hp.from_pretrained).res;
    aval = res.aval;
    bval = res.bval;
    lambda = res.lambda;
end
As = reshape(bases.A*aval,max(bases.N_U)^2,[],hp.num_starting_copies);
Bs = reshape(bases.B*bval,max(bases.N_U)^2,[],hp.num_starting_copies);

if (hp.use_adam)
    ma = zeros(size(aval)); mb = zeros(size(bval));
    va = zeros(size(aval)); vb = zeros(size(bval));
    mlambda = 0; vlambda = 0;
end

% Initialize tracking variables
err_tracker = zeros(hp.num_alts,1);
iter = 0;  % number of updates of A, B, and lambda
iters_per_num_copy = zeros(hp.tot_copies, 1);
iter_this_num_copy = 0;

%%%%%%%%%%%%%%%%%%%%           Training loop           %%%%%%%%%%%%%%%%%%%%
while true
    tic  % start timer for iteration

    % Increase size of description space
    if ((hp.largeB) && (iter == hp.largeB_iter))
        fprintf("Increasing size of B \n");
        % increase B's input dimension to N_U^2
        pi_inc = zeros(hp.n*(hp.k+1), bases.N_U(hp.n)^2);
        for i = 1:bases.N_U(hp.n)
            for j = 1:bases.N_U(hp.n)
                if ((i == 1) && (j ~= 1))
                    pi_inc(j-1, i+bases.N_U(hp.n)*(j-1)) = 1/2;
                end

                if ((i ~= 1) && (j == 1))
                    pi_inc(i-1, i+bases.N_U(hp.n)*(j-1)) = 1/2;
                end
            end
        end
        y_vars = reshape(sdpvar(max(bases.N_U), max(bases.N_U), hp.N, 'symmetric'), max(bases.N_U)^2, hp.N);
        b_vars = sdpvar(size(bases.B2,2),hp.num_copies,'full');
        B_vars = reshape(bases.B2*b_vars,max(bases.N_U)^2,[], hp.num_copies);
        % assign values to b_vars
        Bs = Bs * pi_inc;  % form B matrix to assign new b_vars
        bval = (bases.B2 \ vec(Bs)) + hp.largeB_jolt * randn(size(b_vars,1), 1);
        assign(b_vars, bval);  % assign new b_vars for warm-start
        Bs = reshape(bases.B2*bval,max(bases.N_U)^2,[],hp.num_copies);  % assign new B
        disp(size(y_vars))
    end

    % Get batch for current iteration
    if hp.batch_size < hp.N
        batch_indices = randperm(hp.N, hp.batch_size);
    else
        batch_indices = 1:hp.N;
    end

    %%%%%%%%%%     Alternate 1 - optimize over t, err, y, z      %%%%%%%%%%
    [alt1errs, ~, terminate] = minimize( ...
        iter, batch_indices, Xtrues_train, Xtils_train, fvals_train, ...
        As, Bs, lambda, ...
        t_vars, err_vars, y_vars, z_vars, ...
        hp, bases, ops);

    if (terminate)
        break
    end

    %%%%%%%%%%        Track error and end conditions here        %%%%%%%%%%
    % We break when the iteration count equals the number of times A, B,
    % and lambda are optimized. The error corresponds to the A, B, and
    % lambda parameters after minimizing for `iter` iterations and then
    % fitting the gauge functions. `err_tracker(i)` is the error achieved
    % after `i` training iterations.
    alt1err = sum(alt1errs);
    if contains(hp.approach, "SRV")
        % turned a max to a min optimization problem, flip signs
        alt1err = - alt1err;
    end
    fprintf("iter %d: 1e6 * alt1errsum = %f\n", iter, 1e6 * alt1err)
    if iter > 0
        err_tracker(iter) = 1e6 * alt1err;
        % Terminate when absolute error is below threshold
        if err_tracker(iter) < hp.err_abs_thresh * 1e6
            break
        end
    end
    % Terminate when the number of alternations is reached
    if iter >= hp.num_alts
        break
    end
    % Terminate when the relative change in error has been below threshold
    % for several consecutive iterations
    if iter_this_num_copy > hp.err_consec_iter_bnd
        err_rel_diff = 0;
        for copy = 1:hp.err_consec_iter_bnd
            err_rel_diff = max(err_rel_diff, (err_tracker(iter-copy)-err_tracker(iter-copy+1))/err_tracker(iter-copy));
        end

        % Error increases, occurs when we reach a local optimum and
        % there's variance in machine precision, only for N=batch_size
        % case when error should be monotonically decreasing
        if hp.N == hp.batch_size
            err_inc = err_tracker(iter) > err_tracker(iter-1);
        else
            err_inc = false;
        end

        if (err_rel_diff < hp.err_rel_tol) || (err_inc) % if below threshold, terminate
            if hp.num_copies == hp.tot_copies
                disp("Relative difference too low and reached max copies, exiting.")
                fprintf("err_rel_diff < hp.err_rel_tol: %d < %d\n", err_rel_diff, hp.err_rel_tol);
                fprintf("err_tracker(iter) > err_tracker(iter-1): %d > %d\n", err_tracker(iter), err_tracker(iter-1));
                break
            else
                disp("Relative difference too low, Increasing number of copies \n");
                iters_per_num_copy(hp.num_copies) = iter_this_num_copy;
                iter_this_num_copy = 0;
                hp.num_copies = hp.num_copies + 1;
                % initialize variables of new copy
                aval = randn(size(bases.A,2),1) / (hp.copy_init_magnitude_factor ^ (hp.num_copies - 1));
                bval = randn(size(bases.B,2),1) / (hp.copy_init_magnitude_factor ^ (hp.num_copies - 1));
                Asnc = reshape(bases.A*aval,max(bases.N_U)^2,[],1);
                Bsnc = reshape(bases.B*bval,max(bases.N_U)^2,[],1);
                As = cat(3, As, Asnc);
                Bs = cat(3, Bs, Bsnc);
            end

        end
    end

    % Update iteration trackers
    iter = iter + 1;
    iter_this_num_copy = iter_this_num_copy + 1;

    %%%%%%%%%% Alternate 2 - optimize over t, err, A, B, lambda  %%%%%%%%%%
    [alt2errs, ~, terminate] = minimize( ...
        iter, batch_indices, Xtrues_train, Xtils_train, fvals_train, ...
        A_vars, B_vars, lambda_var, ...
        t_vars, err_vars, value(y_vars), value(z_vars), ...
        hp, bases, ops);
    if (terminate)
        break
    end
    alt2err = sum(alt2errs);
    if contains(hp.approach, "SRV")
        % turned a max to a min optimization problem, flip signs
        alt2err = - alt2err;
    end
    % fprintf("iter %d: 1e6 * alt2errsum = %f\n", iter, 1e6 * alt2err)

    % Update A, B, lambda
    ga = value(a_vars(:,:,1:hp.num_copies)) - aval;
    gb = value(b_vars(:,:,1:hp.num_copies)) - bval;
    glambda = value(lambda_var) - lambda;
    if hp.use_adam
        % Update momentum terms
        ma = hp.beta1 * ma + (1 - hp.beta1) * ga;
        mb = hp.beta1 * mb + (1 - hp.beta1) * gb;
        mlambda = hp.beta1 * mlambda + (1 - hp.beta1) * glambda;
        % Update velocity terms
        va = hp.beta2 * va + (1 - hp.beta2) * ga.^2;
        vb = hp.beta2 * vb + (1 - hp.beta2) * gb.^2;
        vlambda = hp.beta2 * vlambda + (1 - hp.beta2) * glambda.^2;
        % Update parameters
        f = (1 - hp.beta2^iter) / (1 - hp.beta1^iter);
        fprintf("norm of a delta: %f\n", sum(((1 - hp.alpha)^2) * (ma ./ sqrt(va + 1e-8)).^2 * (f ^ 2), 'all'));
        fprintf("norm of b delta: %f\n", sum(((1 - hp.alpha)^2) * (mb ./ sqrt(vb + 1e-8)).^2 * (f ^ 2), 'all'));
        aval = aval + (1 - hp.alpha) * (ma ./ sqrt(va + 1e-8)) * f;
        bval = bval + (1 - hp.alpha) * (mb ./ sqrt(vb + 1e-8)) * f;
        lambda = lambda + (1 - hp.alpha) * (mlambda ./ sqrt(vlambda + 1e-8)) * f;
    else
        fprintf("norm of a delta: %f\n", sum(ga.^2 * ((1-hp.alpha) ^ 2), 'all'));
        fprintf("norm of b delta: %f\n", sum(gb.^2 * ((1-hp.alpha) ^ 2), 'all'));
        aval = aval + (1 - hp.alpha) * ga;
        bval = bval + (1 - hp.alpha) * gb;
        lambda = lambda + (1 - hp.alpha) * glambda;
    end
    As = reshape(bases.A*aval,max(bases.N_U)^2,[],hp.num_copies);
    if ((hp.largeB) && (iter >= hp.largeB_iter))
        Bs = reshape(bases.B2*bval,max(bases.N_U)^2,[],hp.num_copies);
    else
        Bs = reshape(bases.B*bval,max(bases.N_U)^2,[],hp.num_copies);
    end
    lambda = max(lambda, hp.lambda_min);

    toc  % end timer for iteration
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Save error, A_basis, B_basis coefficients, regularization parameter
res = struct();
res.aval = aval;
res.bval = bval;
res.As = As;
res.Bs = Bs;
res.lambda = lambda;
res.err_tracker = err_tracker(1:iter);
res.total_alts = iter;
res.final_error = err_tracker(iter);
res.test_error = -1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end




function [err_vals, sdp_results, terminate] = minimize( ...
    iter, batch_indices, Xtrues, Xtils, fvals, ...
    As, Bs, lambda, ...
    ts, errs, ys, zs, ...
    hp, bases, ops)

    % Check that this is a valid call
    alt1 = ~issdp(As) && ~issdp(Bs) && ~issdp(lambda) &&  issdp(ys) &&  issdp(zs);
    alt2 =  issdp(As) &&  issdp(Bs) &&  issdp(lambda) && ~issdp(ys) && ~issdp(zs);
    assert(issdp(ts), "ts should be a variable.");
    assert(issdp(errs), "errs should be a variable.");
    assert(alt1 ~= alt2, "Check variables and constants in primal-dual constraint.");

    if contains(hp.approach, "SRV")
        optimize_type = "sum";
    else
        optimize_type = "l1";
    end

    vec = @(x) x(:);

    % Track errors
    err_vals = zeros(hp.N, 1);

    if alt2
        % Create constraints over entire data set
        F = [];
        for copy = 1:hp.num_copies
            F = [F, norm(Bs(:,:,copy), 'fro') <= hp.B_bnd];
        end
    end

    % Set up optimization problem and solve
    iii = 0;
    for ii = batch_indices

        iii = iii + 1;  % indices within the batch

        if alt1
            % Reset constraints on each data point
            F = [];
        end

        % primal feasibility
        for copy = 1:hp.num_copies
            F = [F, reshape(As(:,:,copy) * vec(Xtrues(:,:,ii)) + ...
                 Bs(:,:,copy) * vec(ys(:,ii)), bases.N_U(hp.n), bases.N_U(hp.n)) + ...
                 ts(copy,ii) * eye(bases.N_U(hp.n)) >= 0, ...
                 ts(copy,ii) >= 0];
        end

        % dual feasibility
        sum_At_zj = 0; sum_Bt_zj = 0;
        for copy = 1:hp.num_copies
            sum_At_zj = sum_At_zj + As(:,:,copy)' * vec(zs(:,:,copy,ii));
            sum_Bt_zj = sum_Bt_zj + Bs(:,:,copy)' * vec(zs(:,:,copy,ii));
            if alt1
                F = [F, trace(zs(:,:,copy,ii)) <= hp.RW, zs(:,:,copy,ii) >= 0];
            end
        end
        F = [F, norm(sum_Bt_zj, 'fro') <= hp.RW*lambda];
        if alt2
            F = [F, lambda >= hp.lambda_min];
        end
        if contains(hp.approach, "EP")
            F = [F, ...
                    hp.RW*(sum(ts(1:hp.num_copies,ii)) + ...
                    lambda*norm(vec(ys(:,ii)), 'fro')) <= fvals(ii) + errs(iii), ...
                    -sum_At_zj'*vec(Xtrues(:,:,ii)) >= fvals(ii) - errs(iii)
            ];
        elseif hp.approach == "MG"
            F = [F, ...
                norm(Xtrues(:,:,ii)-Xtils(:,:,ii),'fro')^2 / 2 + ...
                hp.RW*(sum(ts(1:hp.num_copies,ii)) + lambda*norm(vec(ys(:,ii)), 'fro')) + ...
                    sum_At_zj'*vec(Xtils(:,:,ii)) + norm(sum_At_zj, 'fro')^2 / 2 <= errs(iii) ...
            ];
        elseif contains(hp.approach, "SRV")
            F = [F, ...
                    hp.RW*(sum(ts(1:hp.num_copies,ii)) + ...
                    lambda*norm(vec(ys(:,ii)), 'fro')) <= fvals(ii), ...
                    sum_At_zj'*vec(Xtils(:,:,ii)) <= errs(iii) ...
                 ];
        else
            error("Unknown approach: %s", hp.approach);
        end

        if alt1
            [sdp_results, terminate] = syl_optimize_wrapper(F, errs(iii), 1, iter, optimize_type, ops);
            err_vals(ii) = value(errs(iii));
        end
    end

    if alt2
        % Return error values
        [sdp_results, terminate] = syl_optimize_wrapper(F, errs, 2, iter, optimize_type, ops);
        err_vals = value(errs);
    end

end


function [issdp_] = issdp(var)
    issdp_ = isa(var, 'sdpvar') || isa(var, 'ndsdpvar');
end