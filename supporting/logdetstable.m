function ld = logdetstable(A, varargin)
    % A stable way to compute the log-determinant
    
    if nargin > 1
        tri = varargin{1};
    else
        tri = false;
    end
        
    %Compute cholesky factorization only if not triangular
    if ~tri
        R = chol(A,'upper');
        ld = 2*sum(log(diag(R)));
    else
        ld = sum(log(diag(A)));
    end
    

end