%% Ex2_dynamic.m
% Driver for synthetic dynamic seismic tomography inverse problem
% where we measure the same angles at all time points and
% where the solution corresponds to two rotating Gaussians.
% Here we consider a Kronecker covariance matrix

% In this case, the hyperparameters are:
% theta = [sigma(noise) alpha(regularization) ell_t ell_s]

clear; close all; clc;

%%% Fix random seed %%%
rng('default')

%% Setup the forward problem
% Generate forward model
nx = 128; ny = 128; nt = 50; ns = 20; nr = 20;   
% nx = 32; ny = 32; nt = 50; ns = 2; nr = 2;   

ProblemOptions = PRset(...
    'phantomImage', 'smooth',... % phantomImage
    'wavemodel','ray',...        % wavemodel - string that defines the type of problem
    's',ns,...                   % s - number of sources in the right side of the domain.
    'p', nr,...                  % p - number of receivers (seismographs)
    'sm',true);                  % sm - logical; if true (default) then A is a sparse matrix, otherwise it is a function handle.
As = PRseismic(nx, ProblemOptions);
At = speye(nt); 
A = kronMat(At,As);

%% Generate data 
% Generate solution
st = rotgauss(nx,ny,nt);

% Generate measurements 
nlevel = 0.02;
b = A*st(:); n = size(b,1); 
[N,sigma] = WhiteNoise(b(:),nlevel);
bn = b + N;

%% Model covariance matrices
% Two dimensional spatial covariance
xmin_s = [0 0];       % Coordinates of left corner
xmax_s = [1 1];       % Coordinates of right corner
nvec_s = [nx, ny];    % Number of points. Here this is a 256 x 256 grid
scale_s = [1.0 1.0];

% One dimensional temporal covariance
xmin_t = 0;           % Coordinates of left corner
xmax_t = 1;           % Coordinates of right corner
nvec_t = nt;          % Number of points. Here this is a 256 x 256 grid
scale_t = 1.0;

%%% Choose hyperprior %%%
prior_type = 'P2'; % Gamma
[~, gradP, logP] = Prior4(prior_type);

%%% Choose kernel %%%
ker_name_t = 'Matern_5by2'; 
ker_name_s = 'Matern_3by2'; 


%% Optimization parameters
Q = priorCovKron(xmin_t, xmax_t, nvec_t, scale_t, xmin_s, xmax_s, nvec_s, scale_s, ker_name_t, ker_name_s);

inv.Q = Q;
inv.A = A;
inv.M = size(A,1);
inv.N = size(A,2);
inv.dn = bn;
inv.prior_type = prior_type;
inv.s_true = st(:);
optim_procedure = 'interior-point';

% Initialize theta. Assume we know the variance of the noise. 
theta_0 = ones(1,4); 
theta_0(1) = sigma^2;

%% Optimizer MC 
% mc = number of Monte Carlo samples
mc = 18; 
inv.mc = mc;    
inv.eps = randn(inv.M, inv.mc);

%% MC with low rank Preconditioner 
inv.precond = 'lowrank';
% nc = number of low rank Cheby points
nc = [10,5]; % space rank, time rank
inv.nc = nc;   

% Precomputation to build the preconditioner
% Offline time
tic
[Uk,~] = lowrank(inv.Q, theta_0, inv.nc);
AUk = [];
for i = 1:size(Uk,2)
  AUk(:,i) = A*Uk(:,i);
end
inv.AUk = AUk;
t_precomp_10_5 = toc

fileid=fopen('iterations.txt','a+'); fprintf(fileid,'With preconditioning 10 5, 1e-7 tol \n'); fclose(fileid);
tic
[t_opt_prec_mc_10_5, fval_10_5, exitflag_10_5, output_10_5, xvals_10_5] = ...
  optimizer(@objfun_mc_symm_eff, theta_0, inv, optim_procedure);
t_mc_opt_10_5 = toc

figure, plot(xvals_10_5), title('MC Prec Reconstructed theta 10 15'), xlabel('iter')


%% MC without Preconditioner 
inv.precond = 'noprecon';

fileid=fopen('iterations.txt','a+'); fprintf(fileid,'Without preconditioning 1e-7 tol \n'); fclose(fileid);

tic,
[t_opt_mc_noprec, fval_noprec, exitflag_noprec, output_noprec, xvals_noprec] = optimizer(@objfun_mc_symm_eff, theta_0, inv, optim_procedure);
t_MC_noprecon = toc
figure, plot(xvals_noprec), title('MC No Prec Reconstructed theta'), xlabel('iter')

%% Display results
name_exp = 'dynamic';

% Plot only these times
index = 1:7:50; 

%% Plot true solution and noisy measurements
xx_true = reshape(st, [nx, ny,nt]);
figure
for i = 1:8
  subplot(2,4,i), imagesc(xx_true(:,:,index(i))), axis off, axis image
  title(sprintf('True i=%d',index(i)))
end

bn_plot= reshape(bn, [ns, nr,nt]);
figure
for i = 1:8
  subplot(2,4,i), imagesc(bn_plot(:,:,index(i))), axis off, axis image
  title(sprintf('Observed i=%d',index(i)))
end

%% Plot reconstrucion using the initial conditions
% Recontructions with initial thetas
[R, ~] = R_mat(theta_0, inv.M);
[mu, ~] = mu_vec(theta_0, inv.N);
[Qm, ~] = Q_mat(inv.Q, theta_0);

% Specify right-hand-side vector for pcg %%%
res = bn - A*mu;
b_test = Qm'*(A'*(R'\res));

%%% Recover signal %%%
tol = 1e-8;
maxit = 100;
[xx_0,~,~,~,~] = pcgRecovery(A,Qm,R,mu,b_test,tol,maxit);

% Calculate Error Between Images
RE = round(norm(xx_0 - st(:),2)/norm(st(:),2),2);
disp(['Initial rel. err = ',num2str(RE)])
xx_0_plot = reshape(xx_0, [nx, ny,nt]);


for i = 1:size(index,2)
  subplot(2,4,i), imagesc(xx_0_plot(:,:,index(i))), axis off, axis image
   title(sprintf('Initial recon i=%d',index(i))) 
 end

%% Plot reconstrucion using MC with PRECONDITIONING
% Recover solution
% theta = [sigma(noise) alpha(regularization) ell_t ell_s] (t_opt_prec_mc)
[R, dR] = R_mat(t_opt_prec_mc_10_5, inv.M);
%R = full(R);
[mu, dmu] = mu_vec(t_opt_prec_mc_10_5, inv.N);
[Qm, dQ] = Q_mat(inv.Q, t_opt_prec_mc_10_5);

% Specify right-hand-side vector for pcg %%%
res = bn - A*mu;
b_test = Qm'*(A'*(R'\res));

%%% Recover signal %%%
tol = 1e-8;
maxit = 100;
[xx_mc,fl0,rr0,it0,rv0] = pcgRecovery(A,Qm,R,mu,b_test,tol,maxit);

% Calculate Error Between Images
RE = round(norm(xx_mc - st(:),2)/norm(st(:),2),2);
disp(['MC LR rel. err = ',num2str(RE)])
xx_mc_plot = reshape(xx_mc, [nx, ny,nt]);

figure
for i = 1:8
  subplot(2,4,i),
    imagesc(xx_mc_plot(:,:,index(i))), axis off, axis image
    title(sprintf('Recon with prec i=%d',index(i)))
 end

%%
% %%% NO PREC
% Recover solution
% theta = [sigma(noise) alpha(regularization) ell_t ell_s] (t_opt_prec_mc)
[R, ~] = R_mat(t_opt_mc_noprec, inv.M);
%R = full(R);
[mu, ~] = mu_vec(t_opt_mc_noprec, inv.N);
[Qm, ~] = Q_mat(inv.Q, t_opt_mc_noprec);

% Specify right-hand-side vector for pcg %%%
res = bn - A*mu;
b_test = Qm'*(A'*(R'\res));

%%% Recover signal %%%
tol = 1e-8;
maxit = 100;
[xx_no_prec,fl0,rr0,it0,rv0] = pcgRecovery(A,Qm,R,mu,b_test,tol,maxit);
% 
% % Calculate Error Between Images
RE = round(norm(xx_no_prec - st(:),2)/norm(st(:),2),2);
disp(['NO PREC LR rel. err = ',num2str(RE)])
xx_no_prec_plot = reshape(xx_no_prec, [nx, ny,nt]);

figure
for i = 1:8
    subplot(2,4,i), imagesc(xx_no_prec_plot(:,:,index(i))), axis off, axis image
    title(sprintf('Recon no prec i=%d',index(i)))
end

%% Plot optimization
figure, plot(xvals_noprec,'LineWidth',3), 
title('MC Reconstructed theta'), xlabel('iter','FontSize',20)
legend('$\sigma$ (noise)', '$\alpha$ (regularization)', '$\ell_t$', '$\ell_s$','Interpreter','latex','FontSize',20)
set(gca,'FontSize',20)


figure, plot(xvals_10_5,'LineWidth',3), 
title('MC Prec Reconstructed theta'), xlabel('iter')
legend('$\sigma$ (noise)', '$\alpha$ (regularization)', '$\ell_t$', '$\ell_s$','Interpreter','latex','FontSize',20)
set(gca,'FontSize',20)
