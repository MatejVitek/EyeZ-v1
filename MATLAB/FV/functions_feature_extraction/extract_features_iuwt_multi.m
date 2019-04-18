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
% This function uses the IUWT (Undecimated Wavelet Transform) to extract
% the vein patterns from the given input images and creating a binary vein 
% output image several times while varying the parameters.
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
function features = extract_features_iuwt_multi(db, roi, roi_small)
%EXTACT_FEATURES_IUWT_MULTI Extract IUWT features using multiple
%parameters
%   Detailed explanation goes here
    global use_progress_bar;
    % Configuration
    % Levels [2,3], [1,2], [1,2,3] and [1,2,3,4] are extracted and fused, see below
    
    % Create the empty features cell array
    [no_users, no_fingers, no_images] = size(db);
    features = cell(no_users, no_fingers, no_images);
    
    if use_progress_bar
        ProgressBar.update('new', 'Extracting IUWT features ...', 'IUWT multi feature extraction');
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
                % IUWT considering levels 1 and 2
                levels=1:2;
                [w, ~] = iuwt_vessel_all(currentImage, levels);
                w_tot12=sum(w,3);
                w_tot12=w_tot12.*currentROI_small;
                bw12=adaptivethreshold(w_tot12,30,0,1);
                bw12=~bw12.*currentROI_small;

                % IUWT considering levels 1,2 and 3
                levels=1:3;
                [w, ~] = iuwt_vessel_all(currentImage, levels);
                w_tot123=sum(w,3);
                w_tot123=w_tot123.*currentROI_small;
                bw123=adaptivethreshold(w_tot123,30,0,1);
                bw123=~bw123.*currentROI_small;

                % IUWT considering levels 1,2,3 and 4
                levels=1:4;
                [w, ~] = iuwt_vessel_all(currentImage, levels);
                w_tot1234=sum(w,3);
                w_tot1234=w_tot1234.*currentROI_small;
                bw1234=adaptivethreshold(w_tot1234,30,0,1);
                bw1234=~bw1234.*currentROI_small;

                % IUWT considering levels 2 and 3
                levels=2:3;
                [w, ~] = iuwt_vessel_all(currentImage, levels);
                w_tot23=sum(w,3);
                w_tot23=w_tot23.*currentROI_small;
                bw23=adaptivethreshold(w_tot23,30,0,1);
                bw23=~bw23.*currentROI_small;

                feat(:,:,1) = bw12;
                feat(:,:,2) = bw123;
                feat(:,:,3) = bw1234;
                feat(:,:,4) = bw23;
                
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

