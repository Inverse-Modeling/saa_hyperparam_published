function [F, gradF, val] = objfun_mc(theta, inv)
%
% [F, gradF, val] = objfun_mc(theta, inv)
%
%   This function computes the objective function and gradient for an
%   approximate marginal posterior distribution, obtained using Monte Carlo
%   samples
%
%      theta - hyperparameters
%      inv   - structure containing information about the problem
%               precond - determines preconditioner, options include
%                             'genGK' and 'lowrank'

if length(theta) ==  3
  [~, gradP, logP] = Prior(inv.prior_type);
elseif length(theta) ==  4
  [~, gradP, logP] = Prior4(inv.prior_type);
end
dn = inv.dn;
A  = inv.A;
hyp_dim = length(theta);


[M,N] = size(A);
[R, dR] = R_mat(theta, M);
[mu, ~] = mu_vec(theta, N);
[Qm, dQ] = Q_mat(inv.Q, theta);

if isequal(class(A),'function_handle')
  A = funMat(@ (x) A(x, 'notransp'), @(x) A(x, 'transp'), A([],'size'));
end
Z = @(x) A*(Qm*(A'*x)) + R*x;
Zmat = funMat(Z, Z, size(R));


% Build preconditioner
switch inv.precond
  case 'genGK'
    genGK_iter = inv.genGK_iter;
    s_true   = inv.s_true;
    %%% reconstruction with fmincon optimal hyperparameters and no regularization %%%
    input = HyBR_lsmrset('RegPar', 1,'x_true', s_true,'Iter',genGK_iter, 'Reorth', 'on');

    % AKS modified to use GenGK rather than GenHyBr
    % [Uk, Bk,~,~] = getGKdecomp(A, Qm, R, dn(:), genGK_iter, input);
    [Uk, Bk, ~] = getGKdecomp2(A, Qm, R, dn(:), genGK_iter, input); % JC faster version
    [G, ldG] = preconditioner(R,Uk,Bk);
  case 'lowrank'
    [~,Mk] = lowrank(inv.Q, theta, inv.nc);

    % % Accuracy of Q
    % Qf = Qm*eye(size(A,2));
    % norm(Qf-Uk*Mk*Uk')/norm(Qf)

    R12 = sqrt(R);
    Uk = R12\inv.AUk;
    [Qk,Rk] = qr(Uk,0);
    Uk = R12*Qk;
    Bk = sqrtm(Rk*Mk*Rk');

[G, ldG] = preconditioner(R,Uk,Bk);

    % Zh = (Uk*Bk)*(Uk*Bk)'+R;

    %     Check errors
    %     Zf = Zmat*eye(M); norm(Zinvres-Zf\res)/norm(Zf\res)
    %     Gg = G*eye(M);
    %     disp('Errors')
    %     norm(Zf-Zh)/norm(Zf)
    %     norm(Zh\eye(M) - Gg'*Gg)/norm(Zh\eye(M))
    %     disp('Determinants')
    %     logdetstable(Zf)
    %     logdetstable(Zh)
    %     2*ldG
case 'noprecon'
        G = 1;
        ldG = 0;
  otherwise
    warning('NotImplemented')
end

res = A*mu - dn;


% logdet(Z) MC estimator
eps = inv.eps;
ns  = size(eps,2);
[ld,lan_iter] = logdetmc(Zmat,G,eps,ldG);

% Implements: Z^{-1}(A*mu-d)
[Zinvres, relres] = krylov_solve(Zmat, G, res, 100, 1.e-6);
lan_iter = lan_iter + length(relres);

% Objective function approximation
F1_k = (1/2)*ld;
F2_k = (1/2)*dot(res,Zinvres);
F = -logP(theta) + F1_k + F2_k;


%%%% Logdet determinant MC estimator for the gradient
Ateps = A'*eps;
Zinveps = zeros(M, ns);
parfor j = 1:ns
  [Zinveps(:,j), relres] = krylov_solve(Zmat, G, eps(:,j), 100, 1.e-6);
  lan_iter = lan_iter + length(relres);
end

AtZinvres = A'*Zinvres;
alpha = zeros(hyp_dim,1);
gamma = zeros(hyp_dim,1);
gradF = zeros(hyp_dim,1);
for i = 1:hyp_dim
  Z_der_eps = A*(dQ{i,1}*Ateps) + dR{i,1}*eps;
  alpha_i = trace(Zinveps'*Z_der_eps)/(2*ns);

  Z_der_res = A*(dQ{i,1}*AtZinvres) + dR{i,1}*Zinvres;
  gamma_i = -(1/2)*dot(Zinvres, Z_der_res);

  alpha(i) = alpha_i; gamma(i) = gamma_i;

  gradF(i,1) = -gradP{i,1}(theta) + alpha_i + gamma_i ;

end
val.dF = [alpha, gamma];
val.lan_iter = lan_iter;

end



function [G,ldG] = preconditioner(R, Uk, Bk)
% Computes a Cholesky-like factorization of
%          Zh = Uk*Bk*Bk'*Uk' + R
% where Uk'*R\Uk = I and Bk is small
% Returns:
% G such that G'*G = Zh^{-1} and computes the logdeterminant of G

R12 = sqrt(R);
[Ub,Sb,~] = svd(Bk, 0);
Uh = R12\(Uk*Ub);
nm = size(R,1);
sb2 = diag(Sb).^2;
I = eye(length(sb2));
D =  I - diag(1./sqrt(1+sb2));

G = funMat( @(x)(R12\x - Uh*(D*(Uh'*(R12\x)))), @(x) R12\(x - Uh*(D*(Uh'*x))),[nm, nm]);
ldG = -logdetstable(R12) + logdetstable(I - D);
end