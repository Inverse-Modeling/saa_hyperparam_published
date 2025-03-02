classdef priorCovKron
  % Prior covariance for Kronecker product
  properties
    xmin_t
    xmax_t
    nvec_t
    pts_t
    scale_t
    xmin_s
    xmax_s
    nvec_s
    scale_s
    pts_s
    ker_name_t
    ker_name_s
    kernel_t
    kernel_s
    gradkernel_t
    gradkernel_s
    theta
  end

  methods
    function Q = priorCovKron(xmin_t, xmax_t, nvec_t, scale_t, xmin_s, xmax_s, nvec_s, scale_s, ker_name_t, ker_name_s)
      % constructor 
      Q.xmin_t = xmin_t;
      Q.xmax_t = xmax_t;
      Q.nvec_t = nvec_t;
      Q.scale_t = scale_t;
      Q.xmin_s = xmin_s;
      Q.xmax_s = xmax_s;
      Q.nvec_s = nvec_s;
      Q.scale_s = scale_s;
      Q.ker_name_t = ker_name_t;
      Q.ker_name_s = ker_name_s;
      [Q.kernel_t,Q.gradkernel_t] = ker_fcn_noscalar_t(ker_name_t);
      [Q.kernel_s,Q.gradkernel_s] = ker_fcn_noscalar_s(ker_name_s);

      [~,Q.pts_t] = createrow(Q.xmin_t, Q.xmax_t, Q.nvec_t, @(r) exp(-r), Q.scale_t);
      [~,Q.pts_s] = createrow(Q.xmin_s, Q.xmax_s, Q.nvec_s, @(r) exp(-r), Q.scale_s);
    end

    function [Qm, dQ] = Q_mat(Q, theta)
      % theta = [sigma(noise) alpha(regularization) ell_t ell_s]
      Q.theta = theta;

      % get dimension of hyperparameters
      r = length(theta);

      % stores values of Q matrix for proper use of 'toeplitzproduct' function
      [Qr_t,~] = createrow(Q.xmin_t, Q.xmax_t, Q.nvec_t, @(r) Q.kernel_t(r, theta), Q.scale_t);
      % computes the matrix-vector product Qx in an efficient manner; see
      % documentation of toeplitzproduct for details
      Qr_t_theta = (theta(2)^2)*Qr_t; % include the scalar
      Qfun_t = @(x) toeplitzproduct(x, Qr_t_theta, Q.nvec_t);
      % stores matrix Q as a funMat object. see documentation of funMat for
      % details
      nt = length(Qr_t_theta);
      Qt = funMat(Qfun_t, Qfun_t, [nt,nt]);

      [Qr_s,~] = createrow(Q.xmin_s, Q.xmax_s, Q.nvec_s, @(r) Q.kernel_s(r, theta), Q.scale_s);
      Qfun_s = @(x) toeplitzproduct(x, Qr_s, Q.nvec_s);
      ns = length(Qr_s);
      Qs = funMat(Qfun_s, Qfun_s, [ns,ns]);
      
      % Since kronMat does not allow multiplication by a scalar, Qt is
      % actually $\theta^2(2)*Q_t(\theta(3))$;
      Qm = kronMat(Qt, Qs);

      % initialization of cell that holds gradients of Q matrix
      dQ = cell(r,1); % stores dQ as a cell of funMat objects
      
      dQ{1} = 0; % Derivative w.r.t. sigma
 
      Qr_t_theta_d = 2*theta(2)*Qr_t; % include the scalar
      Qfun_t_d = @(x) toeplitzproduct(x, Qr_t_theta_d, Q.nvec_t);
      Qt_d = funMat(Qfun_t_d, Qfun_t_d, [nt,nt]);
      dQ{2} =  kronMat(Qt_d, Qs); % Derivative w.r.t. alpha

      % Derivative w.r.t. ell_t
      gradQr_t = createrow(Q.xmin_t,Q.xmax_t,Q.nvec_t,@(r) Q.gradkernel_t{3,1}(r, theta),Q.scale_t);
      gradQr_t_scale = (theta(2)^2)*gradQr_t;
      % computes the matrix-vector product dQx in an efficient manner
      gradQfun_t = @(x) toeplitzproduct(x, gradQr_t_scale, Q.nvec_t);
      Qt_d_scale = funMat(gradQfun_t, gradQfun_t, [nt,nt]);
      dQ{3} = kronMat(Qt_d_scale, Qs);

      % Derivative w.r.t. ell_s
      gradQr_s = createrow(Q.xmin_s,Q.xmax_s,Q.nvec_s,@(r) Q.gradkernel_s{3,1}(r, theta),Q.scale_s);
      gradQfun_s = @(x) toeplitzproduct(x, gradQr_s, Q.nvec_s);
      Qs_d = funMat(gradQfun_s, gradQfun_s, [ns,ns]);
      
      dQ{4} = kronMat(Qt, Qs_d);

      %   %dQ{j} = kronMat(gradQfun_t, Qfun_s) + kronMat(Qfun_t, gradQfun_s);
      %   B = {gradQfun_t,Qfun_t}; C = {Qfun_s, gradQfun_s};
      %   dQ{j} = kronMat(B,C);
      %   % stores dQ as a cell of funMat objects
      %   % dQ{j} = funMat(gradQfun, gradQfun, [n,n]);

    end


    function [U,M] = lowrank(Q, theta, nc)
      % Low rank approx for Q_t

      % [Ut, Mt] = kernel_lowrank_symm(ft, tpts', nc, lims, compress);
      if isscalar(nc)
          nc = [nc,nc];
      end
      sources = Q.pts_t;
      lims.a = Q.xmin_t;   lims.b = Q.xmax_t;
      compress.method = 'nocompress';
      compress.r = min(ceil(nc/2), 10);
      compress.p = 0;
      assert(length(Q.xmin_t) == 1); % Hard coded for 1D but can be adapted appropriately
      ft = @(t1,t2) Q.kernel_t( abs(t2-t1), theta);
      [U_t,M_t] = kernel_lowrank_symm(ft, sources, nc(2), lims, compress);

      % Low rank approx for Q_s
      sources = Q.pts_s;
      lims.a = Q.xmin_s;   lims.b = Q.xmax_s;
      compress.method = 'nocompress';
      compress.r = min(ceil(nc/2), 10);
      compress.p = 0;
      assert(length(Q.xmin_s) == 2); % Hard coded for 2D but can be adapted appropriately
      f = @(x1,x2,x3,x4) Q.kernel_s( sqrt((x3-x1).^2+(x4-x2).^2), theta);
      [U_s,M_s] = kernel_lowrank_symm(f, sources, nc(1), lims, compress);

      M = theta(2).^2*kron(M_t, M_s);
      U = kron(U_t, U_s);
    end

    function [U,M] = lowrank_der(Q, theta, nc)
      sources = Q.pts;
      lims.a = Q.xmin;   lims.b = Q.xmax;
      compress.method = 'nocompress';
      assert(length(Q.xmin) == 2); % Hard coded for 2D but can be adapted appropriately
      M = cell(1,length(theta));
      for j = 1:length(theta)
        f = @(x1,x2,x3,x4) Q.gradkernel{j}( sqrt((x3-x1).^2+(x4-x2).^2), theta);
        [U,M{j}] = kernel_lowrank_symm(f, sources, nc, lims, compress);
      end
    end


  end
end