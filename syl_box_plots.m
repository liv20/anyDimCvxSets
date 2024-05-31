%% Plots the increase in minimum singular value as noise increases

seed = 0; N = 100; n = 2; k = 3; d = 1;
noises = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1];

last_svs = zeros(N, length(noises));
for i = 1:length(noises)
    rng(0)
    [Xtil, ~] = syl_gen_dataset(N, n, k, d, noises(i));
    for ii = 1:N
        M = sylvester(Xtil(1,:,ii), Xtil(2,:,ii));
        svd_ = svd(M);
        last_sv = svd_(length(svd_));
        last_svs(ii,i) = last_sv;
    end
end
last_svs = log10(last_svs);

figure;
boxplot(last_svs, 'Labels', {'0', '1e-4', '1e-3', '1e-2', '1e-1', '1e0'});
ylabel('log of minimum singular value');
xlabel('noise std');
