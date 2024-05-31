function syl_learn(hp, cfg, ops)
% Function: syl_learn
% Description: script to call alternating minimization and evaluate the
%   resulting solution.
%
% Input:
%   - hp: hyperparameters for training
%   - cfg: configuration file for writing plots to a folder and experiments
%       to a spreadsheet
%   - ops: optimization options

%% Validate batch size
assert(0 <= hp.alpha && hp.alpha <= 1)
if (hp.batch_size > hp.N)
    fprintf("Max batch_size is number of points, setting batch size to %d.\n", hp.N);
    hp.batch_size = hp.N;
end
if (ischar(hp.batch_size) && strcmp(hp.batch_size, 'all'))
    hp.batch_size = hp.N;
end
assert(0 <= hp.batch_size)


%% Set up results directory
fprintf("Saving results to %s\n", cfg.fpath);
% Check if the directory already exists
if exist(cfg.fpath, 'dir') == 7
    disp('Directory already exists.');
else
    mkdir(cfg.fpath);
    disp(['Directory "', cfg.fpath, '" created successfully.']);
end


%% Generate data

[Xtils_train, Xtrues_train] = syl_gen_dataset(hp.N, hp.n, hp.k, hp.d, hp.noise);
[Xtils_test, Xtrues_test] = syl_gen_dataset(cfg.test_size, hp.n, hp.k, hp.d, hp.noise);

if contains(hp.approach, "SRV")
    Xtils_train = randn(size(Xtils_train));
    if hp.approach == "SRV"
        Xtils_train = Xtils_train ./ sqrt(sum(Xtils_train.^2, [1,2])) * sqrt(hp.n);
    elseif hp.approach == "SRV_NMR"  % norm max-row
        for ii = 1:hp.N
            Xtils_train(:,:,ii) = Xtils_train(:,:,ii) ./ sqrt(max(sum(Xtils_train(:,:,ii).^2, 2)));
        end
    elseif hp.approach == "SRV_NAR"  % norm all rows to be norm 1
        for ii = 1:hp.N
            for row = 1:hp.n
                Xtils_train(row,:,ii) = Xtils_train(row,:,ii) ./ sqrt(sum(Xtils_train(row,:,ii).^2));
            end
        end
    else
        error("Unknown approach: %s", hp.approach);
    end
end


% EP approach enforces the regularizer to be some value
if ((hp.approach == "EP1") || contains(hp.approach, "SRV"))
   f_vals_train = ones(hp.N, 1);
   f_vals_test = ones(cfg.test_size, 1);
elseif hp.approach == "EPNN"
   f_vals_train = zeros(hp.N, 1);
   f_vals_test = zeros(cfg.test_size, 1);
   for ii = 1:hp.N
       Xtrue = sylvester(Xtrues_train(1,:,ii), Xtrues_train(2,:,ii));
       f_vals_train(ii) = sum(svd(Xtrue));
   end
   for ii = 1:cfg.test_size
       Xtest = sylvester(Xtrues_test(1,:,ii), Xtrues_test(2,:,ii));
       f_vals_test(ii) = sum(svd(Xtest));
   end
else
    f_vals_train = [];
    f_vals_test = [];
end


%% Get bases for linear maps
bases = syl_learn_get_bases(hp, false, [], []);


%% Alternating minimization

successful = true;

if (hp.from_pretrained == "") || (hp.num_alts > 0)
    % Only initialize and train A, B, lambda if beginning from scratch
    % or something needs to train
    try
        res = syl_learn_alt_min(Xtrues_train, Xtils_train, f_vals_train, hp, bases, ops);
    catch ME
        fprintf("Error in alternating minimization.")
        fprintf(ME.message);
        res = struct();
        successful = false;
    end

    % Save training
    save([cfg.fpath 'vars.mat'], "res");

    % Plot 2
    figure;
    if (hp.approach == "MG")
        err_label = 'TSG';  % total sum of gaps
    else
        err_label = 'TSE';  % total sum of errors
    end
    % Plot no-log training curve
    subplot(1,2,1);
    plot(6:length(res.err_tracker), res.err_tracker(6:length(res.err_tracker)), 'b-o', 'LineWidth', 2);
    xlabel('Iterations');
    ylabel([err_label ' x 10^6']);
    title([err_label ': After Iteration 5']);
    % Plot log training curve
    subplot(1,2,2);
    plot(1:length(res.err_tracker), log10(res.err_tracker), 'r-o', 'LineWidth', 2);
    xlabel('Iterations');
    ylabel(['log10 (' err_label ' x 10^6)']);
    title([err_label ': log10 Iterations']);
    grid on;
    set(gcf,'position',[0,0,800,300]);
    saveas(gcf, [cfg.fpath 'err_curves.png']);
else
    res = load(hp.from_pretrained).res;
end


%%%%%%%%%%%%%%%%%%%%            Evaluation             %%%%%%%%%%%%%%%%%%%%
if cfg.eval
% Evaluate on unseen test data
fprintf("Testing final errors\n");

if contains(hp.approach, "EP")
    test_err = 0;
    fprintf("testing final errors\n");
    for jj = 1:cfg.test_size
        x = abs( ...
            f_vals_test(jj) - ...
            syl_evaluate_norm(Xtrues_test(:,:,jj), hp.n, hp.k, ...
                              1, bases.N_U, ...
                              res.lambda, res.As, res.Bs, ops) ...
        );
        fprintf(("final error %d: %d\n"), jj, x);
        test_err = test_err + x;
    end
test_err = 1e6 * sum(test_err);
disp(['Sum of test error = ' num2str(test_err)])

res.test_error = test_err;
end

end


%% Record
fprintf("Row = %d\n", cfg.row);
fields = fieldnames(hp);
values = struct2cell(hp);

alphabet = 'A':'Z';
letters = {};
for letter = alphabet
    letters{end+1} = letter;
end
for letter = alphabet
    letters{end+1} = ['A' letter];
end

writematrix('fpath',cfg.spreadsheet,'Sheet',cfg.worksheet,'Range', [letters{1} '1'])
writematrix(cfg.fpath,cfg.spreadsheet,'Sheet',cfg.worksheet,'Range', [letters{1} num2str(cfg.row)])
for i = 1:numel(fields)
    fieldname = convertCharsToStrings(fields{i});
    fprintf(fieldname);
    fprintf("\n");
    value = values{i};
    writematrix(fieldname,cfg.spreadsheet,'Sheet',cfg.worksheet,'Range', [letters{i+1} '1'])
    writematrix(value,cfg.spreadsheet,'Sheet',cfg.worksheet,'Range', [letters{i+1} num2str(cfg.row)])
end

if (successful)
% Write results too
writematrix("lambda",cfg.spreadsheet,'Sheet',cfg.worksheet,'Range', [letters{i+2} '1'])
writematrix(res.lambda,cfg.spreadsheet,'Sheet',cfg.worksheet,'Range', [letters{i+2} num2str(cfg.row)])
writematrix("total_alts",cfg.spreadsheet,'Sheet',cfg.worksheet,'Range', [letters{i+3} '1'])
writematrix(res.total_alts,cfg.spreadsheet,'Sheet',cfg.worksheet,'Range', [letters{i+3} num2str(cfg.row)])
writematrix("final_error",cfg.spreadsheet,'Sheet',cfg.worksheet,'Range', [letters{i+4} '1'])
writematrix(res.final_error,cfg.spreadsheet,'Sheet',cfg.worksheet,'Range', [letters{i+4} num2str(cfg.row)])
writematrix("test_error",cfg.spreadsheet,'Sheet',cfg.worksheet,'Range', [letters{i+5} '1'])
writematrix(res.test_error,cfg.spreadsheet,'Sheet',cfg.worksheet,'Range', [letters{i+5} num2str(cfg.row)])
end



%% Validation plots
if (successful)
% Boxplots depend on regularization param

if (cfg.plot_comparisons)
figure;
reg_param = 1e-3;

XtruesM = zeros(2*hp.k, 2*hp.k, cfg.test_size);
XtruesSig = zeros(2*hp.k, cfg.test_size);

XtilsM = zeros(2*hp.k, 2*hp.k, cfg.test_size);
XtilsSig = zeros(2*hp.k, cfg.test_size);

XrecsM = zeros(2*hp.k, 2*hp.k, cfg.test_size);
XrecsSig = zeros(2*hp.k, cfg.test_size);

XnucsM = zeros(2*hp.k, 2*hp.k, cfg.test_size);
XnucsSig = zeros(2*hp.k, cfg.test_size);

for ii = 1:cfg.test_size

    XtruesM(:,:,ii) = sylvester(Xtrues_test(1,:,ii), Xtrues_test(2,:,ii));
    XtruesSig(:,ii) = svd(XtruesM(:,:,ii));

    XtilsM(:,:,ii) = sylvester(Xtils_test(1,:,ii), Xtils_test(2,:,ii));
    XtilsSig(:,ii) = svd(XtilsM(:,:,ii));

    Xrec = syl_recover(hp.largeB, hp.n, hp.k, hp.tot_copies, bases.N_U(hp.n), Xtils_test(:,:,ii), ...
        res.lambda, res.As, res.Bs, ops, reg_param);
    XrecsM(:,:,ii) = sylvester(Xrec(1,:), Xrec(2,:));
    XrecsSig(:,ii) = svd(XrecsM(:,:,ii));

    Xnuc = syl_recover_with_nuclear_norm(Xtils_test(:,:,ii), 3e-4);  % lambda
    XnucsM(:,:,ii) = sylvester(Xnuc(1,:), Xnuc(2,:));
    XnucsSig(:,ii) = svd(XnucsM(:,:,ii));

end

XtruesLastSig = reshape(XtruesSig(2*hp.k, :), [], 1);
XtilsLastSig = reshape(XtilsSig(2*hp.k, :), [], 1);
XrecsLastSig = reshape(XrecsSig(2*hp.k, :), [], 1);
XnucsLastSig = reshape(XnucsSig(2*hp.k, :), [], 1);

% Combine the data into a single matrix
combinedData = [log10(XtruesLastSig), log10(XtilsLastSig),...
    log10(XnucsLastSig),  log10(XrecsLastSig)];

% Create two sub-boxplots
subplot(1, 2, 1);  % Subplot for the first column
boxplot(combinedData(:, 1), 'Labels', {'ground truth'});
ylabel('log of smallest singular value');
subplot(1, 2, 2);  % Subplot for the last three columns
boxplot(combinedData(:, 2:end), 'Labels', {'perturbed', 'nuclear', 'recovered'});
sgtitle('magnitude of smallest singular values by recovery method');
% Set figure size and save as a figure
set(gcf, 'Position', [0, 0, 800, 400]);  % Adjust size as needed
saveas(gcf, [cfg.fpath 'boxplots.png']);
end

extended_bases = syl_learn_get_bases(hp, true, res.As, res.Bs);


if (cfg.plot_comparisons)

ns = hp.n:hp.max_n;

% Plot 3 reg vs. off plot
figure;
n_reg_params = 15;
reg_params = logspace(-5, 0, n_reg_params);
reg_params = [reg_params(1), reg_params(9:length(reg_params))];
n_reg_params = length(reg_params);
rec_L2 = zeros([hp.max_n, cfg.test_size, n_reg_params]);
nn_L2  = zeros([hp.max_n, cfg.test_size, n_reg_params]);
rec_sv = zeros([hp.max_n, cfg.test_size, n_reg_params]);
nn_sv  = zeros([hp.max_n, cfg.test_size, n_reg_params]);
eps = 1e-16;

for nn = 1:length(ns)
    fprintf("---------- n = %d ---------- \n", ns(nn))

    % generate new data at a higher dimension
    [Xtils_test, ~] = syl_gen_dataset(cfg.test_size, ns(nn), hp.k, hp.d, hp.noise);

    % restrict extended description to current dimension
    lambda = res.lambda;
    As = extended_bases.psi_Us{ns(nn)}' * extended_bases.A_big * extended_bases.phis{ns(nn)};
    Bs = extended_bases.psi_Us{ns(nn)}' * extended_bases.B_big * extended_bases.phis{ns(nn)};

    for jj = 1:n_reg_params
        fprintf("reg param %d\n", reg_params(jj));
        for ii = 1 : cfg.test_size

            Xrec = syl_recover( ...
                hp.largeB, ns(nn), hp.k, hp.tot_copies, ...
                extended_bases.N_U(ns(nn)), Xtils_test(:,:,ii), ...
                lambda, As, Bs, ops, reg_params(jj));
            % extract last singular value
            % find the log of their geometric means
            for nnn = 1 : ns(nn) - 1
                temp = sylvester(Xrec(nnn,:), Xrec(nnn+1,:));
                temp = svd(temp);
                temp = temp(length(temp));
                rec_sv(nn, ii, jj) = rec_sv(nn, ii, jj) + log10(temp + eps);
            end
            rec_sv(nn, ii, jj) = rec_sv(nn, ii, jj) / (length(ns) - 1);
            rec_L2(nn, ii, jj) = norm(Xrec - Xtils_test(:,:,ii), 'fro');
        
            Xnuc = syl_recover_with_nuclear_norm(Xtils_test(:,:,ii), reg_params(jj));  % lambda
            for nnn = 1 : ns(nn) - 1
                temp = sylvester(Xnuc(nnn,:), Xnuc(nnn+1,:));
                temp = svd(temp);
                temp = temp(length(temp));
                nn_sv(nn, ii, jj) = nn_sv(nn, ii, jj) + log10(temp + eps);
            end
            nn_sv(nn, ii, jj) = nn_sv(nn, ii, jj) / (length(ns) - 1);
            nn_L2(nn, ii, jj) = norm(Xnuc - Xtils_test(:,:,ii), 'fro');
        end
    end

    log10_rec_sv = mean(squeeze(rec_sv(nn,:,:)), 1);
    log10_nn_sv = mean(squeeze(nn_sv(nn,:,:)), 1);
    rec_L2_ = mean(squeeze(rec_L2(nn,:,:)), 1);
    nn_L2_ = mean(squeeze(nn_L2(nn,:,:)), 1);
    
    % Plotting
    plot(log10_rec_sv, rec_L2_, '-s', 'DisplayName', sprintf('ours %d', ns(nn)));
    hold on;
    plot(log10_nn_sv, nn_L2_, '-s', 'DisplayName', sprintf('NN n = %d', ns(nn)));
    xlabel('log10 Singular Value');
    ylabel('L2 Distance Between Noisy and Recovered Matrices')
    legend('show', 'Location', 'southwest');

    grid on;
    
    set(gcf,'position',[0,0,800,400]);
    saveas(gcf, [cfg.fpath sprintf('plot3-n=%d.png', ns(nn))]);
end


end

end

end