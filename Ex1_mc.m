%% Ex1_mc.m
% This script compares the accuracy of the Monte Carlo estimators for the
% static seismic example, for various numbers of Monte Carlo samples.
%
% Results correspond to the left plot in Figure 1 in Section 4 of 
%   "Efficient Sample Average Approximation Techniques for 
%   Hyperparameter Estimation in Bayesian Inverse Problems."

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


Qflag = 2; % flag to specify dimension when generating Q

%%% Add noise to data %%%
[eta,sigma] = WhiteNoise(d, 0.02);
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
ker_name = 'Matern_1by2';
% ker_name = 'Matern_3by2';
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
optim_procedure = 'interior-point';
  
[Fex,~,~] = objfun_full(theta_0,inv);  
 
inv.precond = 'lowrank';
inv.nc = 20; % preconditioner rank
tic
[Uk,Mk] = lowrank(inv.Q, theta_0, inv.nc);
inv.AUk = A*Uk;
tp = toc;
fprintf('Precomputation time is %f.\n', tp)
    

%% Different numbers of Monte Carlo samples, over nr = 10 runs
mclst = 8:16:200;
nr = 10; % number of runs
err = zeros(length(mclst), nr);
for j = 1:length(mclst)
    mc = mclst(j)
    for t = 1:nr
        inv.eps = randn(M, mc);
        [F, gradF, val] = objfun_mc(theta_0, inv);
        err(j,t) = abs(F-Fex)/abs(Fex);
    end
end

%% Plot the relative error as a function of the number of MC samples
figure,
rect = [0,0, 6, 6];
fsize = 12;
set(gcf, 'Units', 'inches');
set(gcf, 'OuterPosition',rect);
set(gcf, 'Position', rect);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'defaultaxesfontsize', fsize);
set(gcf, 'defaulttextfontsize', fsize);
set(0, 'DefaultAxesFontName','Times New Roman');
set(0, 'defaultTextFontName','Times New Roman');
errorbar(mclst,mean(err,2),std(err, 0, 2),'bs-','markersize', 10,'linewidth',2)
ylabel('Relative error', 'FontSize', 24)
set(gca, 'FontSize', 20)
set(gca,'YScale','log')
xlabel('Monte Carlo samples $n_{\rm mc}$', 'FontSize', 24, 'Interpreter', 'LaTeX')

% print -depsc mcrelerr
