function [trp, relres] = bilinearform(A,G,b,f, maxiter,tol,varargin)
    % Compute log(GAG')b using Lanczos approach 
    % 
    % Inputs: 
    % A (n x n) Sparse matrix or funMat type
    % G (n x n) Sparse matrix or funMat type. Preconditioner, such that G'*G = A^{-1} 
    % test {'True', 'False'} Optional parameter. Verifies accuracy of Lanczos relationships
    
    if nargin > 6
        test = varargin{1};
    else
        test = 'False';
    end
    
    n = size(b,1);
    nrmb = norm(b);
    
    %Initialize Lanczos quantities
    V = zeros(n,maxiter);
    T = zeros(maxiter+1,maxiter+1);
    
    %First step 
    vj = b/nrmb;
    vjm1 = b*0;
    beta = 0;
    
    relres = zeros(maxiter,1);
    tr = 0;
    for j = 1:maxiter
        V(:,j) = vj;
        wj = G*(A*(G'*vj)); 
        alpha = wj'*vj;
        wj = wj - alpha*vj -beta*vjm1;
        beta = norm(wj);
        
        %Set vectors for new iterations
        vjm1 = vj;
        vj =    wj/beta;
        
        %Reorthogonalize vj (CGS2) % Change to something more sophisticated
        vj = vj - V(:,1:j)*(V(:,1:j)'*vj);  vj = vj/norm(vj);
        vj = vj - V(:,1:j)*(V(:,1:j)'*vj);  vj = vj/norm(vj);
        
        %Set the tridiagonal matrix
        T(j,j) = alpha;
        T(j+1,j) = beta; T(j,j+1) = beta;
        
        
        %Compute partial Lanczos solution
        Tk = T(1:j,1:j);    Vk = V(:,1:j);
        
        
        [Y,Th] = eig(Tk);   
        omega = Y(1,:);     th = diag(Th);
        trp = sum(omega.^2.*f(th)');
        
        
        %Check differences b/w successive iterations
        relres(j) = abs(trp-tr)/abs(trp);
        if  relres(j) < tol
            relres = relres(1:j);
           
            break
        else
            tr = trp;
  
        end
    end
   
    
    %Test the accuracy of Lanczos
    if strcmp(test,'True')
        
        maxiter = size(relres,1);
        AG = 0*Vk;
        for i = 1:maxiter
                AG(:,i) = A*(G'*Vk(:,i));
        end
        AVk = G*AG;   

        figure, imagesc(log10(abs(Vk'*AVk -Tk))), colorbar
        norm(Vk'*AVk -Tk)
        norm(Vk'*Vk- eye(maxiter))
        ek = zeros(maxiter,1);  ek(end) = beta;
        norm(AVk - Vk*Tk - vj*ek')
    end
end
