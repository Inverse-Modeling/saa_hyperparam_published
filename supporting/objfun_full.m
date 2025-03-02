function [F, gradF, val] = objfun_full(theta, inv)
%
% [F, gradF, val] = objfun_full(theta, inv)
%
%   This function computes the objective function and gradient for the
%   marginal posterior distribution
%      theta - hyperparameters
%      inv   - structure containing information about the problem
%
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
[mu, dmu] = mu_vec(theta, N);
[Qm, dQ] = Q_mat(inv.Q, theta);


%%% Objective Function Computation %%%
if isequal(class(A),'function_handle')
  A = A(eye([M,N]),'notransp');
end
if isa(Qm,'kronMat')
  AA = full(A)';
  tmp = zeros(size(R));
  for i = 1:size(AA,2)
    tmp(:,i) = A*(Qm*AA(:,i));
  end
  Z = R+tmp;
else
  Z = R + A*(Qm*(full(A).'));
end

F1 = (1/2)*logdetstable(Z); % Computes the log determinant term
res = A*mu - dn;
L = Z\res;
%     disp('ZinvRes - Full')
%     L
F2 = (1/2)*dot(res, L); % Computes the residual term

F = - logP(theta) + F1 + F2; % objective function
val.F = [- logP(theta); F1;F2];
val.ZinvRes = L;
val.Z = Z;

%%% Gradient Computation %%%
gradF = nan(hyp_dim,1);
ALPHA = nan(hyp_dim,1);
GAMMA = nan(hyp_dim,1);
DELTA = nan(hyp_dim,1);

for i = 1:hyp_dim
  if isa(Qm,'kronMat')
    % AA = full(A)'; % Already computed
    tmp = zeros(size(R));
    for j = 1:size(AA,2)
      tmp(:,j) = A*(dQ{i,1}*AA(:,j));
    end
    gradZ_t_i = dR{i,1} +tmp;
  else
    gradZ_t_i = dR{i,1} + A*(dQ{i,1}*full(A).');
  end
  k_i = Z\gradZ_t_i;

  alpha_i = (1/2)*trace(k_i); % trace(Z^(-1)*dZ)
  gamma_i = - (1/2)*( (res.')*(k_i*L) ); % (A*mu - dn).'*(Z^(-1)*dZ)*(Z^(-1)*(A*mu - dn))
  delta_i = ( (L.')*(A*dmu{i,1}) ); % (Z^(-1)*(A*mu - dn)).'*A*dmu

  ALPHA(i,1) = alpha_i;
  GAMMA(i,1) = gamma_i;
  DELTA(i,1) = delta_i;

  gradF(i,1) = -gradP{i,1}(theta) + alpha_i + gamma_i + delta_i;
end

val.dF = [ALPHA,GAMMA,DELTA];
%     val.bound = 0;

end