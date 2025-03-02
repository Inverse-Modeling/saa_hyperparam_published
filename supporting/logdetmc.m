function [ld,iter] = logdetmc(A,G,eps, ldG)
%
% [ld,iter] = logdetmc(A,G,eps, ldG)
% This function computes
%     logdet(A) = logdet(GAG') - 2logdet(G)
% where
%     eps contains random vectors
%     ldG = logdet(G)

%Estimate the trace of log(GAG')
tol = 1.e-6;    maxiter = 300;
n = size(G,1);
ns = size(eps,2);

mc = 0;
iter = 0;
parfor i = 1:ns
  [tr,relres] = bilinearform(A,G,eps(:,i),@(x)log(x),maxiter,tol);
  mc = mc + tr;    %eps(:,i)'*xl;
  iter = iter + size(relres,1);
end
mc = n*mc/ns;

%Estimate logdet(A) = logdet(GAG') - 2logdet(G)
ld = mc - 2*ldG;

end
