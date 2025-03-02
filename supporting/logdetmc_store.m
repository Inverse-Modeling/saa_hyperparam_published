function [ld,iter,lanc] = logdetmc_store(A,G, eps, ldG)

    %Compute logdet(G)
    %ldG = logdetstable(G,true);
    
    %Estimate the trace of log(GAG')
    %tol = 1.e-6;    maxiter = 300;
    tol = 1.e-7;    maxiter = 350;
    n = size(G,1);
    ns = size(eps,2); 
    
    mc = 0;
    iter = 0;
    Vklst = cell(ns,1);
    Tklst = cell(ns,1);
    nrmblst = cell(ns,1);
    parfor i = 1:ns
    % for i = 1:ns
       [tr,relres,Vklst{i},Tklst{i},nrmblst{i}] = bilinearform(A,G,eps(:,i),@(x)log(x),maxiter,tol); 
       mc = mc + tr;    %eps(:,i)'*xl;
       iter = iter + size(relres,1);
    end
    mc = n*mc/ns;

    lanc.Vklst = Vklst;
    lanc.Tklst = Tklst;
    lanc.nrmblst = nrmblst;

    
    %Estimate logdet(A) = logdet(GAG') - 2logdet(G)
    ld = mc - 2*ldG;

end


function [trp, relres, Vk, Tk, nrmb] = bilinearform(A,G,b,f, maxiter,tol,varargin)
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
   
    

end
