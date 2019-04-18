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
% image several times while varying the parameters.
%
% Parameters:
%  db        - The input UTFVP images as 3D cell array (60x6x4)
%  roi       - ROI masks corresponding to the input images
%  roi_small - Smaller ROI masks corresponding to the input images
%
% Returns:
%  features - 3D cell array containing the extracted output features. Each
%  cell contains a 3D vector with (image height x image width x number of 
%  extracted features) dimensions. 
% *************************************************************************
function features = extract_features_gb_multi(db, roi, roi_small)
%EXTACT_FEATURES_GB_MULTI Extract Gabor Filter features using multiple
%parameters
%   Detailed explanation goes here
    global use_progress_bar;
    % Get the settings
	global settings;
	ls = settings.FeatureExtractionMultipleSettings.GB;	% local settings
    
    % Create the empty features cell array
    [no_users, no_fingers, no_images] = size(db);
    features = cell(no_users, no_fingers, no_images);
    
    if use_progress_bar
        ProgressBar.update('new', 'Extracting GB features ...', 'GB multi feature extraction');
        starttime = tic;
        nrTotal = no_users*no_fingers*no_images;
    end
    
    for u = 1:no_users
        for f = 1:no_fingers
            for i = 1:no_images
                currentImage = db{u, f, i};
                currentROI = roi{u, f, i};
                currentROI_small = roi_small{u, f, i};
                
                % Feature extraction
                feat=zeros(size(currentImage,1), size(currentImage,2),length(ls.bw)*length(ls.lambda)*length(ls.gamma)*length(ls.sz));
                ft=0;
                for numb=1:length(ls.N)
                    for b=1:length(ls.bw)
                        for l=1:length(ls.lambda)
                            for g=1:length(ls.gamma)
                                for s=1:length(ls.sz)
                                    nn=ls.N(numb);
                                    bb=ls.bw(b);
                                    ll=ls.lambda(l);
                                    gg=ls.gamma(g);
                                    ss=ls.sz(s);
                                    ft=ft+1;
                                    img_out = zeros(size(currentImage,1), size(currentImage,2), nn);

                                    % Rotation of the filter
                                    for n = 1: nn
                                        % orientation of the filter
                                        theta=(n-1)*(pi/nn);
                                        %orien_degree=theta*180/pi;
                                        
                                        % Computation of the kernel of the n-th zero mean even Gabor filter
                                        [ zero_mean_gb ] = zm_gb_filter(ll, bb,gg,ls.p0,theta,ss);
                                        img_out(:,:,n)=imfilter(currentImage, zero_mean_gb, 'symmetric');
                                    end

                                    feat_m = mean(img_out,3).*currentROI_small;

                                    %adaptative thresholding, TODO: parameters externally
                                    binarized_image=adaptivethreshold(feat_m,30,0,1);
                                    BW=bwareaopen(binarized_image,60);
                                    BW=~bwareaopen(~BW,30);
                                    feat(:,:,ft)=BW.*currentROI_small;
                                end
                            end
                        end
                    end
                end
                
                % Store the features
                features{u, f, i} = logical(feat);
                
                if use_progress_bar
                    updateStatus(i + (f-1)*no_images + (u-1)*no_images*no_fingers, nrTotal, toc(starttime));
                end
            end
        end
    end
    
    if use_progress_bar
        ProgressBar.update('close');
    end
end

