function [R, dR, invR] = R_mat(theta, M)
%function [R, dR, invR] = R_mat(theta, M)
%
% This function computes a covariance matrix associated 
%
%                               d = A*s + epsilon
%
%   where
%       -> d       == Known real-valued M-dimensional column vector.
%       -> A       == Known real-valued M-by-N matrix.
%       -> s       == Unknown real-valued N-dimensional column vector.
%       -> epsilon == To be determined Gaussian noise with zero mean
%                     and covariance R: \R^{P}_{+} --> \R^{M x M}. Here
%                     P is the dimension of the hyperparameter space, \R
%                     denotes the real numbers, and \R_{+} denotes the
%                     nonnegative real numbers.
%
%   We assume that s|theta ~ Gauss(mu(),Q()) where
%       -> mu == A function from \R^{P}_{+} to \R^{N} representing the mean
%       -> Q  == A function from \R^{P}_{+} to \R^{N x N} representing the
%                  covariance
%
%  INPUTS
% =========
%   theta - P-dimensional column vector of hyperparameters; all
%           components must be nonnegative
%   M     - dimension of R
%
%  OUTPUTS
% =========
%   R     - Covariance matrix
%   invR  - Inverse of covariance matrix
%   dR    - Gradient of covariance matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version : 2.0 (09-04-2022)
% Author  : Khalil Hall-Hooper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% R matrix
R = theta(1)*speye(M,M);        % covariance associated with noise in data
invR = (1/theta(1))*speye(M,M); % inverse of covariance associated with noise in data
dR{1,1} = speye(M,M);           % gradient of R (w.r.t. theta1)
dR{2,1} = 0*speye(M,M);         % gradient of R (w.r.t. theta2)
dR{3,1} = 0*speye(M,M);         % gradient of R (w.r.t. theta3)
dR{4,1} = 0*speye(M,M);         % gradient of R (w.r.t. theta3)
end

