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
% This function uses the PC (Principal Curvature) to extract the vein
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
function features = extract_features_pc_multi(db, roi, roi_small)
%EXTACT_FEATURES_PC_MULTI Extract Principal Curvature features using multiple
%parameters
%   Detailed explanation goes here
    global use_progress_bar;
    % Get the settings
	global settings;
	ls = settings.FeatureExtractionMultipleSettings.PC;	% local settings
    
    % Create the empty features cell array
    [no_users, no_fingers, no_images] = size(db);
    features = cell(no_users, no_fingers, no_images);
    
    if use_progress_bar
        ProgressBar.update('new', 'Extracting PC features ...', 'PC multi feature extraction');
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
                feat=zeros(size(currentImage,1),size(currentImage,2),length(ls.sigma)*length(ls.thresh));
                j=0;
                for s=1:length(ls.sigma)
                    for t=1:length(ls.thresh)
                        j=j+1;

                        veins_choi = choi_principal_curvature(currentImage,currentROI, ls.sigma(s), ls.thresh(t));
                        veins_choi_bin = kmeans_binarize(veins_choi, currentROI);

                        % clear image, TODO: Parameters externally ???
                        veins_choi_bin = bwareaopen(veins_choi_bin,20);
                        veins_choi_bin=~bwareaopen(~veins_choi_bin,5);
                        feat(:,:,j)=veins_choi_bin;
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

