function [P, gradP, logP] = Prior4(prior_type)
% Hyperprior for 4 parameters
%% Only P1 and P2 work

    if strcmp(prior_type,'P1') % noinformative prior
        % prior and its logarithm
        P = @(theta) 1;
        logP = @(theta) 0;

        % gradient of log(prior)
        gradP1 = @(theta) 0;
        gradP2 = @(theta) 0;
        gradP3 = @(theta) 0;
        gradP4 = @(theta) 0;
        gradP = {gradP1; gradP2; gradP3; gradP4};

    elseif strcmp(prior_type,'P2')  %Gamma prior that is relatively flat

        % prior and its logarithm
        beta = 1.e-4;
        P = @(theta) exp(-beta*sum(theta));
        logP = @(theta) - beta*sum(theta);

        % gradient of log(prior)
        gradP1 = @(theta) -beta;
        gradP2 = @(theta) -beta;
        gradP3 = @(theta) -beta;
        gradP4 = @(theta) -beta;
        gradP = {gradP1; gradP2; gradP3; gradP4};

    elseif strcmp(prior_type,'P3')  % Gamma on theta(1) and theta(2);
        % inverse-Gamma  for theta(3)

        % prior and its logarithm
        beta = 1.e-4;
        delta = 2;
        alpha = 1;
        K = (delta^alpha)/gamma(alpha);
        P1 = @(theta) exp(-beta*sum([theta(1);theta(2)]));
        P2 = @(theta) K*((1/theta(3))^(alpha + 1))*exp(-delta/theta(3));
        P = @(theta) P1(theta)*P2(theta);
        logP = @(theta) log(P(theta));

        % gradient of log(prior)
        gradP1 = @(theta) -beta;
        gradP2 = @(theta) -beta;
        gradP3 = @(theta) -(alpha + 1)/theta(3) + delta/(theta(3)^2);
        gradP = {gradP1; gradP2; gradP3};

    elseif strcmp(prior_type,'P4')  % Gamma on theta(1) and theta(3);
        % inverse-Gamma for theta(2)

        % prior and its logarithm
        beta = 1.e-4;
        delta = 2;
        alpha = 1;
        K = (delta^alpha)/gamma(alpha);
        P1 = @(theta) exp(-beta*sum([theta(1); theta(3)]));
        P2 = @(theta) K*((1/theta(2))^(alpha + 1))*exp(-delta/theta(2));
        P = @(theta) P1(theta)*P2(theta);
        logP = @(theta) log(P(theta));

        % gradient of log(prior)
        gradP1 = @(theta) -beta;
        gradP2 = @(theta) -(alpha + 1)/theta(2) + delta/(theta(2)^2);
        gradP3 = @(theta) -beta;
        gradP = {gradP1; gradP2; gradP3};

    elseif strcmp(prior_type,'P5')  % Gamma on theta(1);
        % inverse-Gamma for theta(2) and theta(3)

        % prior and its logarithm
        beta = 1.e-4;
        delta = 2;
        alpha = 1;
        K = (delta^alpha)/gamma(alpha);
        P1 = @(theta) exp(-beta*theta(1));
        P2 = @(theta) K*((1/theta(2))^(alpha + 1))*exp(-delta/theta(2));
        P3 = @(theta) K*((1/theta(3))^(alpha + 1))*exp(-delta/theta(3));
        P = @(theta) P1(theta)*P2(theta)*P3(theta);
        logP = @(theta) log(P(theta));

        % gradient of log(prior)
        gradP1 = @(theta) -beta;
        gradP2 = @(theta) -(alpha + 1)/theta(2) + delta/(theta(2)^2);
        gradP3 = @(theta) -(alpha + 1)/theta(3) + delta/(theta(3)^2);
        gradP = {gradP1; gradP2; gradP3};

    else
        % display error message
        error('Not an available choice for prior');
    end
end