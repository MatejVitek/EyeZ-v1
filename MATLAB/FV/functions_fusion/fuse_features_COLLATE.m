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
% This function provides a wrapper to use the COLLATE fusion method for
% fusing binary vein images by handling them as segmentations.
%
% Parameters:
%  features    - Input feature 3D cell array containg the features to fuse.
%                Contains a 3D vector with (image height x image width x
%                number of features) dimension at each cell
%  roi_small   - Corresponding ROI masks for the input features
%  prob_values - true if probability values should be used as output
%  varargin    - Optional additional input parameters
%
% Returns:
%  fused_features - 3D cell array containing the fused output features
% *************************************************************************
function fused_features = fuse_features_COLLATE(features, roi_small, prob_values, varargin)
%FUSE_FEATURES_COLLATE Fuse the input features using COLLATE
%   Detailed explanation goes here
    global use_progress_bar;
    % Get the settings
	global settings;
	ls = settings.FusionSettings.COLLATE;	% local settings
    % Create the empty fused features cell array
    [no_users, no_fingers, no_images] = size(features);
    fused_features = cell(no_users, no_fingers, no_images);
    
    if use_progress_bar
        ProgressBar.update('new', 'Fusing features ...', 'COLLATE Fusion');
        starttime = tic;
        nrTotal = no_users*no_fingers*no_images;
    end
    
    for u = 1:no_users
        for f = 1:no_fingers
            for i = 1:no_images
                feat = features{u, f, i};
                fvrn_small = roi_small{u, f, i};
                % Fusion using COLLATE
                % Preparation        
                obs = create_obs('slice', [size(feat, 1) size(feat, 2) 1]);
                R=size(feat,3);
                for r = 1:R
                    obs = add_obs(obs, feat(:,:,r),1,r);
                end
                % Fusion
                [C_est, C_W, ~] = COLLATE(obs, ls.epsilon_C, ls.prior_flag_C, ...
                     ls.init_flag_C, ls.alphas3, ls.cvals3);
                C_est=double(C_est).*fvrn_small;
                C_est=bwareaopen(C_est, ls.areaopen_1);
                C_est=~bwareaopen(~C_est, ls.areaopen_2);
                % Store in cell array
                if ~prob_values
                    fused_features{u, f, i} = C_est;
                else
                    fused_features{u, f, i} = imcomplement(C_W(:,:,:,1,2));
                end
                
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
