%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x_optimal, fval, exitflag, output, xvals] = optimizer(objfunc, theta_0, inv, optim_procedure)
    xvals = [];  % This will contain the values of x where fun has been evaluated
    fun = @(theta) objfunc(theta, inv);
    Aa = [];
    Aeq = [];
    bb = [];
    beq = [];
    hyp_dim = length(theta_0);
    lb = zeros(hyp_dim,1);
    ub = Inf*ones(hyp_dim,1); % original upper bound
    nonlcon = [];
    
    % maxiter = 1e8;
    maxiter = 35;
    % maxiter = 7;
    options = optimoptions(@fmincon,'Algorithm',optim_procedure,...
        'MaxIterations', maxiter, ...
        'MaxFunctionEvaluations', 1e8, ...
        'ConstraintTolerance', 1e-12, ...
        'OptimalityTolerance', 1e-12, ...
        'StepTolerance',1e-14, ...
        'SpecifyObjectiveGradient', true, ...
        'FiniteDifferenceType', 'central', ...
        'Display','iter',...
        'PlotFcn', 'optimplotfval',...
        'CheckGradients', false,...
        'OutputFcn', @outfun);

% [x_optimal,fval,exitflag,output] = fminsearch(fun,theta_0);
%         function stop = outfun(x,optimValues,state)
%             stop=false;
%             if isequal(state,'iter')
%                 xvals = [xvals; x];
%             end
%         end

    [x_optimal,fval,exitflag,output] = fmincon(fun,theta_0,Aa,bb,Aeq,beq,lb,ub,nonlcon,options);
        function stop = outfun(x,optimValues,state)
            stop=false;
            if isequal(state,'iter')
                xvals = [xvals; x];
            end
        end
end