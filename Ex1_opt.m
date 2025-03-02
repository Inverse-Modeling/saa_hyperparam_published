%% Ex1_opt.m
% This script compares the accuracy of the Monte Carlo estimators for the
% static seismic example, at different function evaluations during optimization.
%
% Results correspond to the right plot in Figure 1 and Figure 2 in Section 4 of 
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
inv.genGK_iter = 100;
optim_procedure = 'interior-point';
  
%% Setup preconditioner
inv.precond = 'lowrank';
inv.nc = 20;
mc = 24;
inv.eps = randn(M, mc);
tic
[Uk,Mk] = lowrank(inv.Q, theta_0, inv.nc);
inv.AUk = A*Uk;
tp = toc;
fprintf('Precomputation time is %f.\n', tp)

%% Compute optimization history
tic
[t_opt_mc2, fval4, exitflag4, output4, xvals4] = optimizer(@objfun_mc_symm, theta_0, inv, optim_procedure);
t_MC_LR = toc

%% Compute accuracy at various iterations
nh = size(xvals4,1);
errh = zeros(nh,2);
for i = 1:nh
    theta = xvals4(i,:);
    [Fex,gradex,valex] = objfun_full(theta,inv);

    [F,grad,val] = objfun_mc_symm(theta,inv);

    errh(i,1) = abs(Fex-F)/abs(Fex);
    errh(i,2) = norm(gradex-grad)/norm(gradex);
end

%% Plot the relative error of the objective function as a function of the iteration history
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
semilogy(1:nh, errh(:,1),'bs-','markersize', 10,'linewidth',2)
ylabel('Relative error', 'FontSize', 24)
set(gca, 'FontSize', 20)
set(gca,'YScale','log')
xlabel('iteration', 'FontSize', 24, 'Interpreter', 'LaTeX')
% print -depsc errhist
%% Provide image reconstructions

% Recover image at optimal 
[R, dR] = R_mat(t_opt_mc2, M);
R = full(R);
[mu, dmu] = mu_vec(t_opt_mc2, N);
[Qm, dQ] = Q_mat(inv.Q, t_opt_mc2);

% Specify right-hand-side vector for pcg 
res = d_noise - A*mu;
b = (((res.'/R)*A)*Qm).';
tol = 1e-8;
maxit = 100;
[xx_4,fl0,rr0,it0,rv0] = pcgRecovery(A,Qm,R,mu,b,tol,maxit);

% Calculate Error Between Images
RE = round(norm(xx_4 - s_true,2)/norm(s_true,2),2);
disp(['MC LR rel. err = ',num2str(RE)])
xx_4 = reshape(xx_4, [nx, ny]);

% Recover image at initial guess
[R, dR] = R_mat(theta_0, M);
R = full(R);
[mu, dmu] = mu_vec(theta_0, N);
[Qm, dQ] = Q_mat(inv.Q, theta_0);

tol = 1e-8;
maxit = 100;
[xx_init,fl0,rr0,it0,rv0] = pcgRecovery(A,Qm,R,mu,b,tol,maxit);

% Calculate Error Between Images
RE2 = round(norm(xx_init - s_true,2)/norm(s_true,2),2);
disp(['Init rel. err = ',num2str(RE2)])

%% Image reconstructions
figure,
fsize = 12;
rect = [0,0, 12, 4];
set(gcf, 'Units', 'inches');
set(gcf, 'OuterPosition',rect);
set(gcf, 'Position', rect);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'defaultaxesfontsize', fsize);
set(gcf, 'defaulttextfontsize', fsize);
set(0, 'DefaultAxesFontName','Times New Roman');
set(0, 'defaultTextFontName','Times New Roman');
subplot(1,3,1)
[X,Y] = meshgrid(linspace(0,1,n));
pcolor(X,Y, reshape(s_true, n, n)), shading interp, colorbar
title('Ground truth', 'FontSize', 18,'interpreter','latex')
set(gca, 'FontSize', 16)
axis off
subplot(1,3,2)
pcolor(X,Y, reshape(xx_init, n, n)), shading interp, colorbar
title('Initial $\theta_{0}$', 'FontSize', 18,'interpreter','latex')
set(gca, 'FontSize', 16)
axis off
caxis([0,1])
subplot(1,3,3)
pcolor(X,Y, reshape(xx_4, n, n)), shading interp, colorbar
title('Final $\theta_{\rm prec}$', 'FontSize', 18,'interpreter','latex')
set(gca, 'FontSize', 16)
axis off
caxis([0,1])
 % print -depsc seismic_recon