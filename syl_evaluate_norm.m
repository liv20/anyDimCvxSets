function [val] = syl_evaluate_norm(X, n, k, num_copies, N_U, ...
                                   lambda, As, Bs, ops)
% Function: syl_evaluate_norm
% Description: evaluate regularizer on matrix X
%
% Returns f(X)
% 
% Input:
%   - X: matrix of size n x (k+1)
%   - n: number of Sylvester polynomials per matrix
%   - k: total degree of Sylvester polynomials
%   - num_copies: number of AX+By+tu>0 constraints per data point in
%       regularizer parameterization
%   - N_U: number of monomials from syl_learn_get_bases
%   - lambda: regularization parameter
%   - As, Bs: regularizer parameters
%   - ops: optimization options
% Output:
%   - val: f(X)
    y = sdpvar(n*(k+1), 1, 'full');
    t = sdpvar(1, num_copies, 'full');
    objective = sum(t) + lambda * norm(y, 'fro');

    F = [];
    for copy = 1:num_copies
        F = [F, reshape(As(:,:,copy) * X(:) + Bs(:,:,copy) * vec(y), N_U(n), N_U(n)) + t(:, copy) * eye(N_U(n)) >= 0];
        F = [F, t(:, copy) >= 0];
    end
    optimize(F, objective, ops);
    
    val = value(objective);
end
