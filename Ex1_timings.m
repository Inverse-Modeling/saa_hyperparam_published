%% Ex1_timings.m
% This script compares timings for the Monte Carlo estimators for the
% static seismic example, for various numbers of measurements.
%
% Results correspond to Section 4 of 
%   "Efficient Sample Average Approximation Techniques for 
%   Hyperparameter Estimation in Bayesian Inverse Problems."

clear; close all; clc;

%%% Fix random seed %%%
rng('default')

%% Problem initialization
theta_0 = [1.e-03, rand(1,2)];

n = 2^7;
% n = 2^8;

for j = 1:3  
    ProblemOptions = PRset(...
        'phantomImage', 'smooth',... % phantomImage
        'wavemodel','ray',...        % wavemodel - string that defines the type of problem
        's',32*j,...                   % s - number of sources in the right side of the domain.
        'p', 45*j,...                  % p - number of receivers (seismographs)
        'sm',true);                  % sm - logical; if true (default) then A is a sparse matrix, otherwise
    %      it is a function handle.
    [A, d, s_true, ProblemInfo] = PRseismic(n, ProblemOptions);

    fprintf('Measurement Size is %5d\n',prod(ProblemInfo.bSize))
    d = d/n;
    A = A/n;
    M = dot(size(A),[1,0]);
    N = dot(size(A),[0,1]);


    Qflag = 2; % flag to specify dimension when generating Q

    %%% Add noise to data %%%
    level = 0.02; % noise percentage/level
    eta = randn(size(d(:)));
    nN = norm(eta(:));
    eta = eta / nN;

    eta = level*norm(d(:))*eta;
    sigma = level*norm(d(:))/nN;

    d_noise = d + eta;

    %%% Choose problem domain %%%
    %%% 2D Setup %%%
    xmin = [0 0];             % Coordinates of left corner
    xmax = [1 1];             % Coordinates of right corner
    nx = n;
    ny = n;
    nvec = [nx ny];             % Number of points in grid
    scale = [1.0 1.0];        % Parameters governing length scales.

    %%% Choose prior %%%
    prior_type = 'P2'; % Gamma
    [~, gradP, logP] = Prior(prior_type);

    %%% Choose kernel %%%
    % ker_name = 'Matern_1by2';
    ker_name = 'Matern_3by2';
    % ker_name = 'Matern_5by2';


    [kernel,gradkernel] = ker_fcn(ker_name);
    Q = priorCov(xmin, xmax, nvec, scale, ker_name);
    inv.Q = Q;
    inv.A = A;
    inv.M = M;
    inv.N = N;
    inv.dn = d_noise;
    inv.prior_type = prior_type;
    inv.s_true = s_true;
    inv.genGK_iter = 100;
    optim_procedure = 'interior-point';
    

    mc = 24;
    inv.eps = randn(M, mc);
    inv.precond = 'lowrank';
    inv.nc = 15;
    tic
    [Uk,Mk] = lowrank(inv.Q, theta_0, inv.nc);
    inv.AUk = A*Uk;
    tp = toc;
    fprintf('Precomputation time is %f.\n', tp)
    
    tic
    [F, gradF, val] = objfun_mc(theta_0, inv);
    tc = toc;
    fprintf('MC Computation time is %f.\n', tc)
    fprintf('Avg Number of iterations is %g.\n',val.lan_iter/mc)
    
    if j < 3
        tic
        [Fex, gradF, val] = objfun_full(theta_0, inv);
        tc = toc;
        fprintf('Full Computation time is %f.\n', tc)
        fprintf('Err is %e.\n', abs(F-Fex)/abs(Fex)); 
    end
    disp('--------------------------------------')
end


