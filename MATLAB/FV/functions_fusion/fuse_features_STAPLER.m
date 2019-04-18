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
% This function provides a wrapper to use the STAPLER fusion method for
% fusing binary vein images by handling them as segmentations and
% constructing a virtual ground-truth using log odds majority voting first.
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
function fused_features = fuse_features_STAPLER(features, roi_small, prob_values, varargin)
%FUSE_FEATURES_STAPLER Fuse the input features using STAPLER
%   Detailed explanation goes here
    global use_progress_bar;
	% Get the settings
	global settings;
	ls = settings.FusionSettings.STAPLER;	% local settings
    % Create the empty fused features cell array
    [no_users, no_fingers, no_images] = size(features);
    fused_features = cell(no_users, no_fingers, no_images);
    
    if use_progress_bar
        ProgressBar.update('new', 'Fusing features ...', 'STAPLER Fusion');
        starttime = tic;
        nrTotal = no_users*no_fingers*no_images;
    end
    
    for u = 1:no_users
        for f = 1:no_fingers
            for i = 1:no_images
                feat = features{u, f, i};
                fvrn_small = roi_small{u, f, i};
                % Fusion using STAPLER
                % Preparation
                % Use MV for creating a "truth" image
                feat_mean= sum(feat,3);
                t=floor((size(feat,3)+1)/2);
                feat_fin=zeros(size(feat, 1), size(feat, 2));
                feat_fin(feat_mean>=t)=1;
                feat_fin=feat_fin.*fvrn_small;
                % clean the image
                feat_fin=bwareaopen(feat_fin, ls.areaopen_2);
                feat_fin=~bwareaopen(~feat_fin, ls.areaopen_1);
                truth=feat_fin;
                
                obs = create_obs('slice', [size(feat, 1) size(feat, 2) 1]);
                R=size(feat,3);
                for r = 1:R
                    obs = add_obs(obs, feat(:,:,r),1,r);
                end

                [~, prior] = log_odds_majority_vote(obs, 1);
                svprior=prior;
                % Fusion
                [bias_theta, ~] = construct_theta_bias(obs, truth, obs, ls.cf1);
                [SR_est, SR_W, ~] = STAPLER(obs, ls.epsilon_S, svprior, ...
                    ls.init_flag_S, ls.cf1, bias_theta);
                SR_est=double(SR_est).*fvrn_small;
                SR_est=bwareaopen(SR_est, ls.areaopen_1);
                SR_est=~bwareaopen(~SR_est, ls.areaopen_2);
                % Store in cell array
                if ~prob_values
                    fused_features{u, f, i} = SR_est;
                else
                    fused_features{u, f, i} = SR_W(:,:,1,2); 
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
