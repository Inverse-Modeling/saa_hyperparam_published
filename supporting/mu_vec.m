function [mu, dmu] = mu_vec(theta, N)
%function [mu, dmu] = mu_vec(theta, N)
%
% This function computes a mean vector associated with 
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
%   N     - dimension of mu
%
%  OUTPUTS
% =========
%   mu   - Mean vector
%   dmu  - Gradient of mean vector

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version : 2.0 (09-04-2022)
% Author  : Khalil Hall-Hooper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mu = zeros(N,1);        % mu vector

dmu{1,1} = 0*sparse(N,1); %zeros(N,1);  % gradient of mu (w.r.t. theta1)
dmu{2,1} = 0*sparse(N,1); %zeros(N,1);  % gradient of mu (w.r.t. theta2)
dmu{3,1} = 0*sparse(N,1); %zeros(N,1);  % gradient of mu (w.r.t. theta3)
dmu{4,1} = 0*sparse(N,1); %zeros(N,1);  % gradient of mu (w.r.t. theta3)
end

