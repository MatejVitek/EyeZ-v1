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
% This function extracts all the six binary features (MC, PC, WLD, RLT, GF
% and IWUT) for later use with the second type of fusion (including
% different feature extractors). The parameters for all the feature
% extractors can be set at the beginning of the function.
%
% Parameters:
%  db   - Preprocessed images of the UTFVP dataset (60x6x4 cell array)
%  roi  - Corresponding ROI masks for the UTVFP images
%  roi1 - Corresponding smaller ROI masks for the UTVFP images
%
% Returns:
%  feat_mc   - Extracted MC (Maximum Curvature) features
%  feat_gb   - Extracted GF (Gabor Filter) features
%  feat_wld  - Extracted WLD (Wide Line Detector) features
%  feat_pc   - Extracted PC (Principal Curvature) features
%  feat_rlt  - Extracted RLT (Repeated Line Tracking) features
%  feat_iuwt - Extracted IUWT (Undecimated Wavelet Transform) features
% *************************************************************************
function [feat_mc, feat_gb, feat_wld, feat_pc, feat_rlt, feat_iuwt] = extractSingleFeatures(db, roi, roi1)
%EXTRACTSINGLEFEATURES Extracts the six binary vein features for use with
%the second type of fusion
%   Detailed explanation goes here
    global use_progress_bar;
    % Get the settings
	global settings;
	ls = settings.FeatureExtractionSingleSettings;
    
    %% Load the features
    [no_users, no_fingers, no_images]=size(db);

    feat_mc     = cell(no_users, no_fingers, no_images);
    feat_gb     = cell(no_users, no_fingers, no_images);
    feat_wld    = cell(no_users, no_fingers, no_images);
    feat_pc     = cell(no_users, no_fingers ,no_images);
    feat_rlt    = cell(no_users, no_fingers, no_images);
    feat_iuwt   = cell(no_users, no_fingers, no_images);
    
    if use_progress_bar
        ProgressBar.update('new', 'Extracting all features ...', 'Single feature extraction');
        starttime = tic;
        nrTotal = no_users*no_fingers*no_images;
    end

    %% Do the feature extraction
    for p=1:no_users
        for f=1:no_fingers
            for m=1:no_images
                disp(strcat('Extracting features subject id: ',num2str(p),', finger id: ', num2str(f),', image id: ', num2str(m)))
                
                finger_image=db{p,f,m};
                currentROI=roi{p,f,m};
                currentROI_small=roi1{p,f,m};

                % MC
                v_max_curvature = miura_max_curvature(finger_image, currentROI, ls.mc_sigma);
                md = median(v_max_curvature(v_max_curvature>0));
                v_max_curvature_bin = v_max_curvature > md;
                feat_mc{p,f,m} = v_max_curvature_bin;

                % RLT
                v_repeated_line = miura_repeated_line_tracking(finger_image, currentROI, ls.rlt_max_iterations, ls.rlt_r, ls.rlt_W);
                md = median(v_repeated_line(v_repeated_line>0));
                v_repeated_line_bin = v_repeated_line > md;
                feat_rlt{p,f,m} = v_repeated_line_bin;   

                % PC   
                veins_choi = choi_principal_curvature(finger_image, currentROI, ls.pc_sigma, ls.pc_thresh);
                veins_bin = kmeans_binarize(veins_choi, currentROI).*currentROI;
                veins_bin=bwareaopen(veins_bin, ls.pc_bw_thres);
                feat_pc{p,f,m}=logical(veins_bin);

                % WLD
                veins_huang = huang_wide_line(uint8(finger_image.*255), currentROI, ls.wld_r1, ls.wld_t1, ls.wld_g1);
                feat_wld{p,f,m} = veins_huang;
                
                % IWUT
                [w, ~] = iuwt_vessel_all(finger_image, ls.iuwt_levels);
                w_tot23=sum(w,3);
                w_tot23=w_tot23.*currentROI_small;
                bw=adaptivethreshold(w_tot23, ls.iuwt_bin_thres, 0, 1);
                bw=~bw.*currentROI_small;            
                BW=bwareaopen(bw, ls.iuwt_bw_thres);
                BW=~bwareaopen(~BW, ls.iuwt_bw_thres);
                feat_iuwt{p,f,m}=BW;
                
                % GB
                feat_gb{p,f,m}=gabor_feat_extract(finger_image, currentROI, ls.gb_p0, ls.gb_N, ls.gb_bw, ls.gb_lambda, ls.gb_gamma, ls.gb_sz);
                
                if use_progress_bar
                    updateStatus(m + (f-1)*no_images + (p-1)*no_images*no_fingers, nrTotal, toc(starttime));
                end
            end
        end
    end
    
    if use_progress_bar
        ProgressBar.update('close');
    end
end

