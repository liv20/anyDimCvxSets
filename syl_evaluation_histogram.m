%% Plots the distribution of regularizer evaluations
% evaltrues: matrices with approximate GCD structure
% evaltils: perturbed matrices
% evalXonS: matrices on the sphere (squared Frobenius norm of n)
% evalXrowsl2:  matrices in which the L2 norm of each row is 1

% Load regularizer parameter values from here!
res = load("runs/run_2024-05-29_10-08-50/vars.mat").res;

% may need to change based on where the solution is loaded from
hp = struct( ...
    "impose_ext", true, ...
    "N", 100, ...
    "n", 2, ...
    "k", 3, ...
    "d", 1, ...
    "noise", 1e-2, ...
    "delta", 1e-10);

ops = sdpsettings('solver','mosek','verbose', 0, ...
                  'mosek.MSK_DPAR_INTPNT_CO_TOL_DFEAS', hp.delta, ...
                  'mosek.MSK_DPAR_INTPNT_CO_TOL_PFEAS', hp.delta, ...
                  'mosek.MSK_DPAR_INTPNT_CO_TOL_MU_RED', hp.delta, ...
                  'mosek.MSK_DPAR_INTPNT_TOL_DFEAS', hp.delta, ...
                  'mosek.MSK_DPAR_INTPNT_CO_TOL_REL_GAP', hp.delta);

bases = syl_learn_get_bases(hp, false, [], []);

[Xtrues, Xtils] = syl_gen_dataset(hp.N, hp.n, hp.k, hp.d, hp.noise);

XonS = randn(size(Xtrues));
XonS = XonS ./ sqrt(sum(XonS.^2, [1,2])) * sqrt(hp.n);

Xrowsl2 = randn(size(Xtrues));
for ii = 1:100
    Xrowsl2(:,:,ii) = Xrowsl2(:,:,ii) ./ sqrt(sum(Xrowsl2(:,:,ii).^2, 2));
end

evaltrues = zeros(hp.N,1);
evaltils = zeros(hp.N,1);
evalXonS = zeros(hp.N,1);
evalXrowsl2 = zeros(hp.N,1);
for ii = 1:100
    evaltrues(ii) = syl_evaluate_norm(Xtrues(:,:,ii),2,3,1,bases.N_U,res.lambda,res.As,res.Bs,ops);
    evaltils(ii) = syl_evaluate_norm(Xtils(:,:,ii),2,3,1,bases.N_U,res.lambda,res.As,res.Bs,ops);
    evalXonS(ii) = syl_evaluate_norm(XonS(:,:,ii),2,3,1,bases.N_U,res.lambda,res.As,res.Bs,ops);
    evalXrowsl2(ii) = syl_evaluate_norm(Xrowsl2(:,:,ii),2,3,1,bases.N_U,res.lambda,res.As,res.Bs,ops);
end


data = [evaltrues, evaltils, evalXonS, evalXrowsl2];
% Create a boxplot
figure;
boxplot(data, 'Labels', {'evaltrues', 'evaltils', 'evalXonS', 'evalXrowsl2'});
ylabel('Values');

figure;
% Plot first histogram
subplot(2,2,1);
histogram(evaltrues);
title('evaltrues');
% Plot second histogram
subplot(2,2,2);
histogram(evaltils);
title('evaltils');
% Plot third histogram
subplot(2,2,3);
histogram(evalXonS);
title('evalXonS');
% Plot fourth histogram
subplot(2,2,4);
histogram(evalXrowsl2);
title('evalXrowsl2');
shg

figure;
