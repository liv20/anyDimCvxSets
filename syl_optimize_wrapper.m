function [results, terminate] = syl_optimize_wrapper(F, errors, part, ~, optimize_type, ops)
% Function: optimize_wrapper
% Description: wrapper for evaluating optimization problems
% 
% Input:
%   - F: list of constraints
%   - errors: sdpvar whose sum or norm should be minimized
%   - part: part during alternating minimization, for debugging purposes
%   - optimize_type: if "sum", minimize sum; otherwise, minimize L1 norm
%   - ops: optimization options
% Output:
%   - results: error code
%   - terminate: true for nonzero error code

if (length(errors) == 1)
    results = optimize(F, errors, ops);
else
    if optimize_type == "sum"
        results = optimize(F, sum(errors, 'all'), ops);
    else  % default use L1 norm
        results = optimize(F, norm(errors, 1), ops);
    end
end

if results.problem == 4    % numerical problems (rare)
    warning('Init. caused numerical problems (part %d)', part)
elseif results.problem == 2
    warning('Unbounded objective function (part %d)', part)
elseif results.problem ~= 0 % other problems
    error(['Unknown error with YALMIP code (part %d) ' num2str(results.problem) ' ' yalmiperror(results.problem)], part)
end
terminate = results.problem ~= 0;
end
