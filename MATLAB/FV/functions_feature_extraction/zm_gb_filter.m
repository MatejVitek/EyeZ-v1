% *************************************************************************
% This file is part of the feature level fusion framework for finger vein
% recognition (MATLAB implementation). 
%
% Reference:
% Advanced Variants of Feature Level Fusion for Finger Vein Recognition 
% C. Kauba, E. Piciucco, E. Maiorana, P. Campisi and A. Uhl 
% In Proceedings of the International Conference of the Biometrics Special 
% Interest Group (BIOSIG'16), pp. 1-12, Darmstadt, Germany, Sept. 21 - 23
%
% Authors: Christof Kauba <ckauba@cosy.sbg.ac.at> and 
%          Emanuela Piciucco <emanuela.piciucco@stud.uniroma3.it>
% Date:    31th August 2016
% License: Simplified BSD License
%
% 
% Description:
% This function is an implementation of a zero mean Gabor filter
%
% Parameters:
%  lambda    - wavelength of Gabor filter (lambda >= 2) 
%  bw        - bandwidth of the filter    (generally bw = 1)
%  gamma     - spatial aspect ratio
%  p0        - =[x0,y0]' translation of origin (2x1 vector)
%  theta     - angle of rotation of the filter
%  sz        - dimension of the filter 
%
% Returns:
%  zero_mean_gb - Array with the corresponding Gabor filter kernel
% *************************************************************************
function [ zero_mean_gb ] = zm_gb_filter(lambda, bw, gamma, p0, theta, sz)

% Implementation of zero mean Gabor filter
% Gabor filter is defined as follow:
%                 1
% h(p)=   --------------*cos[wm'(p1-p01)]*exp(-0.5*(p1-p01)'*C^-1*(p1-p01))
%          2*pi*|C|^1/2
% where p=[x,y]', p0 = [x0,y0]', wm=[pi/2a,0]' C = sigma_x.^2,0;0,sigma_y.^2] 
% and p1 and p01 are the coordinates after the trasformation obtained
% by the trasformation matrix [cos(theta), sen(theta); -sen(theta), cos(theta)]

sigma = lambda/pi*sqrt(log(2)/2)*(2^bw+1)/(2^bw-1); 
sigma_x = sigma;        % standard deviation of the Gaussian part in the horizontal direction (x axes)
sigma_y = sigma/gamma;  % standard deviation of the Gaussian part in the vertical direction (y axes)



% spatial modulation frequency
wm=[2*pi/lambda,0]';  

% % sz: dimension of the filter 
% alternatively, use a fixed size
% sz = 60;
% sz=fix(8*max(sigma_y,sigma_x));

if mod(sz,2)==0
    sz=sz+1;
end


[x y]=meshgrid(-fix(sz/2):fix(sz/2),fix(sz/2):-1:fix(-sz/2));
x0= p0(1).*ones(sz,sz);
y0= p0(2).*ones(sz,sz);

% Rotation of coordinates
x_theta=x*cos(theta)+y*sin(theta);
y_theta=-x*sin(theta)+y*cos(theta);

x0_theta=x0*cos(theta)+y0*sin(theta);
y0_theta=-x0*sin(theta)+y0*cos(theta);

x_theta1=x_theta-x0_theta;
y_theta1=y_theta-y0_theta;

% Gaussian part
% K=1/(2*pi*sigma_x*sigma_y);
 %gauss_part=K.*exp(-0.5*(x_theta1.^2/sigma_x^2+y_theta1.^2/sigma_y^2));
gauss_part=exp(-0.5*(x_theta1.^2/sigma_x^2+y_theta1.^2/sigma_y^2));

% Cosinusoidal part
psi = -wm(1).* x0_theta;
cos_part=cos(wm(1)*x_theta1+psi);

gb= gauss_part.*cos_part;

m=mean2(gb);
zero_mean_gb = gb-m.*ones(size(gb,1),size(gb,2));

end