
function [Xrec] = syl_recover_with_nuclear_norm(Xtil, reg_param)
% Function: syl_recover_with_nuclear_norm
% Description: use nuclear norm regularization to recover a low-rank matrix
% from the perturbed matrix Xtil
%
% Returns argmin_X ||X - Xtil||^2 + lambda ||X||_*
% 
% Input:
%   - Xtil: perturbed matrix of size n x (k+1)
%   - reg_param: regularization strength
% Output:
%   - Xrec: recovered matrix of size n x (k+1)

    n = size(Xtil, 1);
    k = size(Xtil, 2) - 1;

    F = [];
    nuc_obj = 0;

    ps = sdpvar(n, k+1, 'full');
    Y = sdpvar(n-1, 2*k, 2*k, 'full');
    for row = 1 : n - 1
        for ii = 1:k
            Y(row, ii, ii:ii+k) = ps(row);
            Y(row, ii, 1:ii-1) = 0;
            Y(row, ii, ii+k+1:2*k) = 0;
            Y(row, ii+k, ii:ii+k) = ps(row+1);
            Y(row, ii+k, 1:ii-1) = 0;
            Y(row, ii+k, ii+k+1:2*k) = 0;
        end

        W1 = sdpvar(2*k, 2*k);
        W2 = sdpvar(2*k, 2*k);
        nuc_obj = nuc_obj + (trace(W1) + trace(W2)) / 2;
        M = [W1, squeeze(Y(row,:,:)); squeeze(Y(row,:,:)).', W2];
        F = [F, M >= 0];
    end

    objective = norm(ps - Xtil, 'fro') ^ 2 + reg_param * nuc_obj;

    ops = sdpsettings('solver','mosek','verbose', 0);
    optimize(F, objective, ops);
    Xrec = value(ps);

end