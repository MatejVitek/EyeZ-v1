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
% This function uses the GF (Gabor Filter) to extract the vein
% patterns from the given input images and creating a binary vein output
% image.
%
% Parameters:
%  img       - The input vein image
%  fvr       - The binary ROI mask to use
%  p0        - =[x0,y0]' translation of origin (2x1 vector)
%  N         - number of orientations (different angles theta) to use
%  bw        - bandwidth of the filter    (generally bw = 1)
%  lambda    - wavelength of Gabor filter (lambda >= 2) 
%  gamma     - spatial aspect ratio
%  sz        - dimension of the filter 
%  
% Returns:
%  feature_gb - output binarized vein image containing the output of the
%  Gabor filtered input vein image (vein pattern)
% *************************************************************************
function [ feature_gb ] = gabor_feat_extract(img,fvr,p0,N,bw,lambda,gamma,sz)

img_out = zeros(size(img,1), size(img,2), N);

%% Rotation of the filter
%figure;
for n = 1: N
    % orientation of the filter
    theta=(n-1)*(pi/N);
    %orien_degree=theta*180/pi;
    
    % Computetion of the kernel of the n-th zero mean even Gabor filter
    [ zero_mean_gb ] = zm_gb_filter(lambda, bw,gamma,p0,theta,sz);
    
    img_out(:,:,n)=imfilter(img, zero_mean_gb, 'symmetric');
    
    %   subplot(4,N/4,n); imshow(img_out(:,:,n)); %title(strcat('Image filtered with kernel with orientation:', ' ', num2str(orien_degree),' degrees'))
    %    imshow(zero_mean_gb/2+0.5);
    %    title(strcat('Orientation:', ' ', num2str(orien_degree),' degrees'))
    
end

% Do morphological post processing and binarization
feat = mean(img_out,3).*fvr;
binarized_image=adaptivethreshold(feat,30,0,1);
BW=bwareaopen(binarized_image,60);
BW=~bwareaopen(~BW,30);
feature_gb=~BW.*fvr;
feature_gb = logical(feature_gb);
% figure;
% subplot(1,2,1); imshow(adapthisteq(finger_image./255),[]);
% subplot(1,2,2); imshow(BW);


% %% Percentage thresholding
% 
% proportion=0.45;
% [ binarized_image ] = percentage_thesholding( feat, roi, proportion );
% binarized_image=binarized_image.*roi;
% figure; imshow(binarized_image);




end
