function st = rotgauss(nx,ny,nt)
    x = linspace(0,1,nx);
    y = linspace(0,1,ny);
    
    [X,Y] = meshgrid(x,y);
    st = zeros(nx*ny,nt);
    
%     theta = linspace(0,2*pi,nt);
theta = linspace(0,pi/2,nt);
%     theta = fliplr(theta);
    for i = 1:nt
        %First bump
        xc = 0.5 + 0.25*cos(theta(i));  yc = 0.5 + 0.25*sin(theta(i));
        R1 = exp(-(X-xc).^2./(0.2.^2) - (Y-yc).^2./(0.2.^2));
        
        %Second bump
        xc = 0.5 + 0.25*cos(theta(i)+4*pi/3);  yc = 0.5 + 0.25*sin(theta(i)+ 4*pi/3);
        R2 = exp(-(X-xc).^2./(0.2.^2) - (Y-yc).^2./(0.2.^2));
        
        st(:,i) = R1(:) + R2(:);
    end
    
    %st = 0.5*(1+sign(st-0.65)); 
end