function veins = choi_principal_curvature(img,fvr,sigma,thresh)
% Parameters:
%  img    - Input vascular image
%  fvr    - Finger vein region
%  sigma  - Final sigma applied
%  thresh - Percentage of maximum used for hard thresholding 

% Returns:
%  veins - Vein image

% Reference:
% Finger vein extraction using gradient normalization and principal
%   curvature
% J.H. Choi, W. Song, T. Kim, S.R. Lee and H.C. Kim
% Image Processing: Machine Vision Applications II
% Proc. SPIE 7251, 725111 (2009)
% doi: 10.1117/12.810458

% Author:  Bram Ton <b.t.ton@alumnus.utwente.nl>
% Date:    13th March 2012
% License: This work is licensed under a Creative Commons
%          Attribution-NonCommercial-ShareAlike 3.0 Unported License

sigma = sqrt(sigma^2/2);

gx = ut_gauss(img,sigma,1,0);
gy = ut_gauss(img,sigma,0,1);
Gmag = sqrt(gx.^2 + gy.^2); %  Gradient magnitude

% Apply threshold
gamma = (thresh/100)*max(max(Gmag));
indices = find(Gmag < gamma);
gx(indices) = 0;
gy(indices) = 0;

% Normalise
Gmag( find(Gmag == 0) ) = 1; % Avoid dividing by zero
gx = gx./Gmag;
gy = gy./Gmag;

hxx = ut_gauss(gx,sigma,1,0);
hxy = ut_gauss(gx,sigma,0,1);
hyy = ut_gauss(gy,sigma,0,1);

lambda1 = 0.5*(hxx + hyy + sqrt(hxx.^2 + hyy.^2 -2*hxx.*hyy + 4*hxy.^2));
veins = lambda1.*fvr;

% Normalise
veins = veins + -1*min(veins(:));
veins = veins/max(veins(:));

veins = veins.*fvr;