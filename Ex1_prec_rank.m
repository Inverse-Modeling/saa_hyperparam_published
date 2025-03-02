%% Ex1_prec_rank.m
% This script compares the accuracy of the Monte Carlo estimators for the
% static seismic example, for various Matern kernels and for various
% preconditioner ranks.
%
% Results correspond to Table 1 in Section 4 of 
%   "Efficient Sample Average Approximation Techniques for 
%   Hyperparameter Estimation in Bayesian Inverse Problems."
% 
clear; close all; clc;

%%% Fix random seed %%%
rng('default')

%% Problem initialization
theta_0 = [1.e-03, rand(1,2)];
n = 2^7;
% n = 2^8;
ProblemOptions = PRset(...
    'phantomImage', 'smooth',... % phantomImage
    'wavemodel','ray',...        % wavemodel - string that defines the type of problem
    's',32,...                   % s - number of sources in the right side of the domain.
    'p', 45,...                  % p - number of receivers (seismographs)
    'sm',true);                  % sm - logical; if true (default) then A is a sparse matrix, otherwise

[A, d, s_true, ProblemInfo] = PRseismic(n, ProblemOptions);

d = d/n;
A = A/n;
M = dot(size(A),[1,0]);
N = dot(size(A),[0,1]);

%%%% Prior covariance
xmin = [0 0];             % Coordinates of left corner
xmax = [1 1];             % Coordinates of right corner
nx = n;
ny = n;
nvec = [nx ny];             % Number of points in grid
scale = [1.0 1.0];        % Parameters governing length scales.
Qflag = 2; % flag to specify dimension when generating Q

%%% Add noise to data %%%
level = 0.02; % noise percentage/level
[eta,sigma] = WhiteNoise(d, level);
d_noise = d + eta;

%%% Choose problem domain %%%
inv.A = A;
inv.M = M;
inv.N = N;
inv.dn = d_noise;

inv.s_true = s_true;
inv.precond = 'lowrank';
mc = 24;
inv.eps = randn(M, mc);
optim_procedure = 'interior-point';

%%% Choose prior %%%
prior_type = 'P2'; % Gamma
[~, gradP, logP] = Prior(prior_type);
inv.prior_type = prior_type;

%%% Choose kernel and number of Chebyshev points%%%
ker_name = {'Matern_1by2', 'Matern_3by2', 'Matern_5by2'};
nc = 5:5:20;

%% Compute error and the iteration
err = zeros(3, length(nc));
iter = zeros(3, length(nc));
for i = 1:3
    [kernel,gradkernel] = ker_fcn(ker_name{i});
    Q = priorCov(xmin, xmax, nvec, scale, ker_name{i});
    inv.Q = Q;
    [Fex,gradex,valex] = objfun_full(theta_0,inv);  
        
        
    %%Setup preconditioner%%
    for j = 1:length(nc)
        inv.nc = nc(j);
        
        [Uk,Mk] = lowrank(inv.Q, theta_0, inv.nc);
        inv.AUk = A*Uk;
         
        [F,grad,val] = objfun_mc_symm(theta_0,inv);

        err(i,j) = abs(Fex-F)/abs(Fex);
        iter(i,j) = val.lan_iter/mc;
    end
end

%% Print Table
disp('Accuracy')
for i = 1:size(err,1)
  fprintf(ker_name{i})
  for j = 1:size(err,2)
    fprintf(' %.4e ', err(i,j))
  end
  fprintf('\n')
end

fprintf('\n')
disp('Average Lanczos iterations')
for i = 1:size(err,1)
  fprintf(ker_name{i})
  for j = 1:size(err,2)
    fprintf(' %.2f ', iter(i,j))
  end
  fprintf('\n')
end