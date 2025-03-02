function [x_recov,flag,relres,iter,resvec] = pcgRecovery(A,Q,R,mu,b,tol,maxit)
%%% funtion handle specifying left-hand-side matrix for pcg %%%
fcn = @(x) lhs_pcg(x,A,Q,R);

%%% use pcg to solve for Q\x; then solve for x %%%
[Qinv_x_recov,flag,relres,iter,resvec] = pcg(fcn,b,tol,maxit);
x_recov = mu + Q*Qinv_x_recov;
end

function A_pcg = lhs_pcg(x,A,Q,R)
%A_pcg = lhs_pcg(x,mu,A,Q,R)  Left-hand-side Ahat matrix in matrix equation
%                             Ahat*(Q\x) = b
%
% A function handle specifying the matrix-vector product
%
%                           Ahat*(Q\x)
%
% where, Ahat = Q*A.'*(R\A)*Q + Q
%
%  INPUT(S)
% =========
%   x      - real-valued vector; size(x) = [N,1];
%   mu     - real-valued vector; size(x) = [N,1];
%   A      - real-valued matrix; size(A) = [M,N];
%   Q      - symmetric positive definite matrix; size(Q) = [N,N];
%   R      - symmetric positive definite matrix; size(R) = [M,M].
%
%
%  OUTPUT(S)
% =========
%   Ahat   - A symmetric positive definite matrix; size(Ahat) = [N;N].
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version : 1.0 (05-10-2022)
% Author  : Khalil Hall-Hooper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y = Q*x;
A_pcg = Q*(A'*(R\(A*y))) + y;
end

