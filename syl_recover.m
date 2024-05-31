function [Xhat, primal_value] = syl_recover(largeB, n, k, num_copies, N_U, ...
                                            Xtil, lambda, As, Bs, ops, reg_param)
% Function: syl_recover
% Description: use any-dimensional convex sets regularizer to recover
% a low-rank matrix from the perturbed matrix Xtil
%
% Returns argmin_X ||X - Xtil||^2 + lambda f(X)
% 
% Input:
%   - largeB: true if B_n is an element of U_n rather than W_n; false
%       otherwise
%   - n: number of Sylvester polynomials per matrix
%   - k: total degree of Sylvester polynomials
%   - num_copies: number of AX+By+tu>0 constraints per data point in
%       regularizer parameterization
%   - N_U: number of monomials from syl_learn_get_bases
%   - Xtil: perturbed matrix of size n x (k+1)
%   - lambda: regularization parameter
%   - As, Bs: regularizer parameters
%   - ops: optimization options
%   - reg_param: regularization strength
% Output:
%   - Xhat: recovered matrix of size n x (k+1)
%   - primal_value: value of the recovery problem
    Xhat = sdpvar(n, k+1, 'full');
    if (largeB)
        y = reshape(sdpvar(N_U, N_U, 'symmetric'), N_U^2, 1);
    else
        y = sdpvar(n*(k+1), 1, 'full');
    end
    t = sdpvar(1, num_copies, 'full');
    objective = norm(Xhat - Xtil, 'fro')^2 / 2 + ...
               reg_param * (sum(t) + lambda * norm(vec(y), 'fro'));

    F = [];
    for copy = 1:num_copies
        F = [F, reshape(As(:,:,copy) * Xhat(:) + Bs(:,:,copy) * y, N_U, N_U) + t(:, copy) * eye(N_U) >= 0];
        F = [F, t(:, copy) >= 0];
    end
    
    optimize(F, objective, ops);
    primal_value = value(objective);
    Xhat = value(Xhat);
end