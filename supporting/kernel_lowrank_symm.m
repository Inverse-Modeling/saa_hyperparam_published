function [U,M] = kernel_lowrank_symm(kernel, sources, n, lims, compress)
    % Input
    % 
    % kernel : function of the form f(x_1,...,x_N, y_1, ... y_N)
    % sources: Ns x N source points
    % n      : Number of Chebyshev points in each dimension
    % lims   : data structure  with the bounding boxes in each mode
    % compress: data structure containing information for the compression
    
    % Output
    % 
    % U, M: matrices which approximate the kernel interactions
     

    N = size(sources, 2);
    
    % Chebyshev interpolation of a 2N dimensional function
    a = [lims.a, lims.a]; b = [lims.b, lims.b];
    xpts = [sources, sources];
    [F, S] =  multi_cheb(kernel, a, b, n, xpts); % issue?
%     disp(['class(F) = ', class(F)])
%     disp(['class(S) = ', class(S)])

    % Compress the tensor using the desired method
    switch compress.method
        case 'nocompress'
            % Do nothing

        case 'randhosvd'
            ell = compress.r+compress.p;
            U = cell(1,2*N);
            for j = 1:N
                Fj = tens2mat(F, j);
                Yj = Fj*randn(size(Fj,2), ell);
                [Uj,~] = qr(Yj,0);
                
                U{j}   = Uj;
                U{j+N} = Uj;
            end
            F = tmprod(F, U, 1:(2*N), 'T');
            
            for j = 1:(2*N); S{j} = S{j}*U{j}; end
    end
    
    % Factor matrices; compute rowwise Khatri-Rao product in reverse order
    Ulst = cell(1,N);   
    for j = 1:N
        Ulst{N-j+1} = S{j}';
    end
    U = kr(Ulst)'; % issue?
%     disp(['class(U) = ', class(U)])
    
    % Reshape the core tensor using mode unfolding
    M = tens2mat(F, 1:N); 
    
    
end


function [F, S] = multi_cheb(f, a, b, n, xpts)
    
    % Order of the tensor
    N = length(a);
    
    % Generate Chebyshev grid and compute the function values
    xs = cell(N,1);
    for j = 1:N
        xs{j} = mapfromcheby(cos((2*(n:-1:1)-1)*pi/(2*n)), a(j), b(j));
       
    end
    [Xs{1:N}] = ndgrid(xs{:});
    
    % Select function values
    F = f(Xs{:});
    
    % Factor matrices
    S = cell(N, 1);
    for j = 1:N
        S{j} = Sn(maptocheby(xpts(:,j), a(j), b(j)), maptocheby(xs{j}, a(j), b(j)), n);
    end
    
    
end

function xcheb = maptocheby(x,a,b) %to [-1,1]
    xcheb = 2*(x-a)./(b-a) -1;
end

function xreal = mapfromcheby(x,a,b) %to [a,b]
    xreal = (x+1)*(b-a)/2 + a;
end

function S = Sn(x,y,n)
  
  Tx = @(x,n) cos(n*acos(x));
  
  S = 1/n;
  for k = 1:n-1
    S = S + (2/n)*Tx(x,k).*Tx(y,k);
  end
  
end
