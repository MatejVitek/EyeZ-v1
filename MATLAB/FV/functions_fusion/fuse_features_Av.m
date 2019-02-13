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
% This function implements the Averaging fusion scheme. This version
% does not support weighting of the single feature inputs (weighting has to
% be applied before calling the function), for weighting support use the 
% FeatureFusion.m which is a wrapper for all fusion methods.
%
% Parameters:
%  features    - Input feature 3D cell array containg the features to fuse.
%                Contains a 3D vector with (image height x image width x
%                number of features) dimension at each cell
%  varargin    - Optional additional input parameters (not used)
%
% Returns:
%  fused_features - 3D cell array containing the fused output features
% *************************************************************************
function fused_features = fuse_features_Av(features, varargin)
%FUSE_FEATURES_Av Fuse the input features using average (mean)
%   Detailed explanation goes here
    global use_progress_bar;
    % Create the empty fused features cell array
    [no_users, no_fingers, no_images] = size(features);
    fused_features = cell(no_users, no_fingers, no_images);
    
    if use_progress_bar
        ProgressBar.update('new', 'Fusing features ...', 'Average (Mean) Fusion');
        starttime = tic;
        nrTotal = no_users*no_fingers*no_images;
    end
    
    for u = 1:no_users
        for f = 1:no_fingers
            for i = 1:no_images
                % Fusion using mean
                feat = features{u, f, i};
                fused_features{u, f, i} = mean(feat,3);
                
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
