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
% This function uses the RLT (Repeated Line Tracking) to extract the vein
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
function features = extract_features_rlt_multi(db, roi, roi_small)
%EXTACT_FEATURES_RLT_MULTI Extract Repeated Line Tracking features using multiple
%parameters
%   Detailed explanation goes here
    global use_progress_bar;
    % Get the settings
	global settings;
	ls = settings.FeatureExtractionMultipleSettings.RLT;	% local settings
    
    % Create the empty features cell array
    [no_users, no_fingers, no_images] = size(db);
    features = cell(no_users, no_fingers, no_images);
    
    if use_progress_bar
        ProgressBar.update('new', 'Extracting RLT features ...', 'RLT multi feature extraction');
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
                feat=zeros(size(currentImage,1),size(currentImage,2),length(ls.max_iterations)*length(ls.r)*length(ls.W));
                j=0;
                for m=1:length(ls.max_iterations)
                    for rr=1:length(ls.r)
                        for ww=1:length(ls.W)
                            j=j+1;
                            v_repeated_line = miura_repeated_line_tracking(currentImage, currentROI, ls.max_iterations(m),ls.r(rr),ls.W(ww));
                            md = median(v_repeated_line(v_repeated_line>0));
                            v_repeated_line_bin = v_repeated_line > md;

                            feat(:,:,j)=v_repeated_line_bin;
                        end
                    end
                end
                
                % Store the features
                features{u, f, i} = feat;
                
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

