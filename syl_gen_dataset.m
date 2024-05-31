function [Xtils, Xtrues] = syl_gen_dataset(N, n, k, d, noise)
% Function: syl_gen_dataset
% Description: generate N matrices containing n Sylvester polynomials of
%  degree k
% Input:
%   - N: number of example matrices in the dataset
%   - n: number of Sylvester polynomials per matrix
%   - k: total degree of Sylvester polynomials
%   - d: degree of the generator polynomial
%   - noise: level of noise in polynomial
% Output:
%   - Xtil: dataset (indexed by 3rd column) of perturbed matrices
%   - Xtrue: dataset of true matrices
%   - both Xtil and Xtrue have size n x (k+1) x N

    % Create datasets
    Xtils = zeros(n, k+1, N); Xtrues = zeros(n, k+1, N);

    for i = 1:N
        % generate random polynomial g of degree d
        g = randn(1,d+1);
        g = g / norm(g,2);
        
        % Generate random polynomials q and fill the matrix
        for j = 1:n
            q = randn(1,k-d+1);
            
            % x = g * q 
            x = zeros(1,k+1);
            for c = 1:k+1
                xc = 0;
                for c2 = max(1, c+d-k):min(d+1, c)
                    xc = xc + g(c2) * q(c-c2+1);
                end
                x(c) = xc;
            end

            % Store in data
            Xtils(j, :, i) = x + noise * randn(1, k+1);
            Xtrues(j, :, i) = x;

            % normalize rows of data
            Xtils(j, :, i) = Xtils(j, :, i) / norm(Xtils(j, :, i));
            Xtrues(j, :, i) = Xtrues(j, :, i) / norm(Xtrues(j, :, i));
        end
    end
end