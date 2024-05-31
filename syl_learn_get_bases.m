function [bases] = syl_learn_get_bases(hp, get_extended_bases, A, B)
% Function: syl_learn_get_bases
% Description: Return bases and embeddings for U and V
% Input:
%   - hp: hyperparameters, contains parameters for the data-generation
%       process
%   - get_extended_bases: if true, requires A and B; returns bases for
%       As and Bs in a higher-dimensional description space
%   - A, B: only required if get_extended_bases is false
% Output:
%   - bases: a struct containing bases for A, B and their embedding
%       functions

vec = @(x) x(:);

k_desc = 1;

if ~get_extended_bases
%% Set up description spaces
% We start with description spaces V = W = R^{n x (k+1)}
% U = Sym^2( Sym{\leq k_desc}(V) )
% W will be extended to W2 = U
d_V = 1; d_W = 1;
d_U = 2 * k_desc;

[~, K, KW2] = get_bases(hp.n, hp.k, k_desc);


% generate embeddings
phis = cell(hp.n, 1); psi_Us = cell(hp.n, 1);
N_U = zeros(hp.n, 1); % number of monomials for each dim.
for ii = 1:hp.n
    [phis{ii}, psi_Us{ii}] = get_embeddings(hp.n, ii, hp.k);
    N_U(ii) = ii*(hp.k+1)+1;
end

% Add extendability conditions:
if hp.impose_ext
    for ii = 1:d_V
        % Ensure A extends to a morphism
        K = [K; kron(phis{ii}', speye(N_U(end)^2) - psi_Us{ii}*psi_Us{ii}')];
    end
    for ii = 1:d_U
        % Ensure A' extends to a morphism
        K = [K; kron(speye(hp.n*(hp.k+1)) - phis{ii}*phis{ii}', psi_Us{ii}')];
    end

    for ii = 1:d_W
        KW2 = [KW2; kron(psi_Us{ii}', speye(N_U(end)^2) - psi_Us{ii}*psi_Us{ii}')];
    end
    for ii = 1:d_W
        KW2 = [KW2; kron(speye(N_U(end)^2) - psi_Us{ii}*psi_Us{ii}', psi_Us{ii}')];
    end
end

% Find bases for kernels:
[~,SpRight] = spspaces(K,2); Abasis = SpRight{1}(:, SpRight{3});
[~,SpRight] = spspaces(K,2); Bbasis = SpRight{1}(:, SpRight{3});
[~,SpRight] = spspaces(KW2,2); B2basis = SpRight{1}(:, SpRight{3});

bases = struct();
bases.A = Abasis;
bases.B = Bbasis;
bases.B2 = B2basis;
bases.phis = phis;
bases.psi_Us = psi_Us;
bases.N_U = N_U;  % vector


else
%% Extend description to higher dimension

[N_U_m, K_m, ~] = get_bases(hp.max_n, hp.k, k_desc);

% Form embeddings
[phi_m, psi_U_m] = get_embeddings(hp.max_n, hp.n, hp.k);

phis = cell(hp.max_n, 1); psi_Us = cell(hp.max_n, 1);
N_U = zeros(hp.max_n, 1); % number of monomials for each dim.
for ii = 1 : hp.max_n
    [phis{ii}, psi_Us{ii}] = get_embeddings(hp.max_n, ii, hp.k);
    N_U(ii) = ii*(hp.k+1)+1;
end

% Extend A, B by solving linear systems
A_big = lsqr([K_m; kron(sparse(phi_m)', psi_U_m')],sparse([zeros(size(K_m,1),1); vec(A)]), 2e-16, 1e4);
A_big = reshape(A_big, N_U_m^2, []);

B_big = lsqr([K_m; kron(sparse(phi_m)', psi_U_m')],sparse([zeros(size(K_m,1),1); vec(B)]), 2e-16, 1e4); 
B_big = reshape(B_big, N_U_m^2,[]);

bases = struct();
bases.A_big = A_big;
bases.B_big = B_big;
bases.N_U = N_U;  % scalar
bases.phis = phis;
bases.psi_Us = psi_Us;
bases.phi_m = phi_m;
bases.psi_U_m = psi_U_m;

end

end


function [N_U, K, KW2] = get_bases(m, k, k_desc)
% Helper function to syl_learn_get_bases

% get generators for S_n
Pi = zeros(m, m, 2);
Pi(:,:,1) = eye(m); Pi(:,[1,2],1) = Pi(:,[2,1],1);
Pi(:,:,2) = eye(m); Pi(:,:,2) = Pi(:, [m,1:m-1],2);

% get generators for S_2
D = zeros(k+1, k+1, 1);
D(:,:,1) = diag((-1) .^ (0:k));

% get action of generators on R^{m*(k+1)}
Pib = zeros(m*(k+1), m*(k+1), 3);
Pib(:,:,1) = kron(eye(k+1), Pi(:,:,1));
Pib(:,:,2) = kron(eye(k+1), Pi(:,:,2));
Pib(:,:,3) = kron(D, eye(m));

% get induced action of generators on polynomials in n*(k+1) variables
x_ext = sdpvar(m*(k+1),1);
m_U = monolist(x_ext, k_desc);            % monomials of degree <= k_U in n*(k+1) variables
deg_list = get_deg_list(m_U, x_ext); % multi-degrees for each monomial
N_U = size(deg_list,1);              % size of matrices in U

Pi_U = gen_algebra_map(Pib, x_ext, deg_list); % get action of generators

%% Get bases for linear maps
% generate matrices whose kernels are spaces of extendable, equivariant
% linear maps

K = []; KW2 = []; % K: V -> U, KW2: W2 -> U
for ii = 1:size(Pi,3)
    % action of generators on symmetric matrices index by monomials:
    G = kron(sparse(Pi_U(:,:,ii)), sparse(Pi_U(:,:,ii)));

    % append equations for equivariance
    K = [K; kron(sparse(Pib(:,:,ii))', speye(N_U^2)) - kron(speye(m*(k+1)), G)];
    KW2 = [KW2; kron(speye(N_U^2),G) - kron(G',speye(N_U^2))];
end

% Since we represent symmetric matrices as full matrices, require linear
% maps to map symmetric matrices to symmetric matrices and act by zero on
% skew-symmetric matrices.
Tperm_U = gen_transpose_perm_mtx(N_U); % permutation matrices sending vec(X) to vec(X')
K = [K; kron(speye(m*(k+1)),Tperm_U) - speye(m*(k+1)*N_U^2)];
KW2 = [KW2; kron(speye(N_U^2), Tperm_U) - speye(N_U^4)];
KW2 = [KW2; kron(Tperm_U',speye(N_U^2)) - speye(N_U^4)];

end




function [phi, psi] = get_embeddings(n, n_0, k)
% get embeddings V_{n_0} to V_n and U_{n_0} to U_n
% Inputs:
%   - n:    number of rows in X
%   - n_0:  starting number of rows in X
%   - k:    k in U = Sym^2( Sym{\leq k}(V)
% Outputs:
%   - phi:  embedding for V
%   - psi:  embedding for U

    % embed R^{n_0*(k+1)} to R^{n*(k+1)}
    id = eye(n);
    phi = id(:,1:n_0); phi = kron(eye(k+1), phi);
    % create block diagonal for phi_ext
    phi_ext = blkdiag(1, phi);
    psi = kron(phi_ext, phi_ext);
    % store as sparse matrices
    phi = sparse(phi); psi = sparse(psi);
end