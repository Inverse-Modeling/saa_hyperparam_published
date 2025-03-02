function [kernel,gradkernel] = ker_fcn(ker_name)

if strcmp(ker_name,'Matern_1by2') == true
    % kernel function
    kernel = @(r, theta) (theta(2)^2).*exp(-r/theta(3));
    
    % gradient of kernel function
    gradK1 = @(r, theta) 0*r;
    gradK2 = @(r, theta) 2*theta(2)*exp(-r/theta(3));
    gradK3 = @(r, theta) (r/(theta(3)^2))*(theta(2)^2).*exp(-r/theta(3));
    gradkernel = {gradK1; gradK2; gradK3};
    
elseif strcmp(ker_name,'Matern_3by2') == true
    % kernel function
    kernel = @(r, theta) (theta(2)^2).*(1 + ...
        (r*sqrt(3))/theta(3) ).*exp(-(r*sqrt(3))/theta(3));
    
    % gradient of kernel function
    gradK1 = @(r, theta) 0*r;
    gradK2 = @(r, theta) (2*theta(2)).*(1 + ...
        (r*sqrt(3))/theta(3) ).*exp(-(r*sqrt(3))/theta(3));
    gradK3 = @(r, theta) (theta(2)^2).*( ...
        (3*(r.^2))/(theta(3)^3)).*exp(-(r*sqrt(3))/theta(3));
    gradkernel = {gradK1; gradK2; gradK3};
    
elseif strcmp(ker_name,'Matern_5by2') == true
    % kernel function
    kernel = @(r, theta) (theta(2).^2).*(1 + (r.*sqrt(5))/theta(3) + (5.*(r.^2))/(3.*(theta(3).^2)) ).*exp(-(r.*sqrt(5))/theta(3));
    
    % gradient of kernel function
    gradK1 = @(r, theta) 0*r;
    gradK2 = @(r, theta) (2*theta(2)).*(1 + (r*sqrt(5))/theta(3) + (5*(r.^2))/(3*(theta(3).^2)) ).*exp(-(r*sqrt(5))/theta(3));
    gradK3 = @(r, theta) (theta(2).^2).*( ((5*sqrt(5))/3).*(r.^3/(theta(3).^4)) + (5/3)*(r.^2/(theta(3).^3)) ).*exp(-(r*sqrt(5))/theta(3));
    gradkernel = {gradK1; gradK2; gradK3};
    
else
    % display error message
    error('Not an available choice for kernel');
end
end