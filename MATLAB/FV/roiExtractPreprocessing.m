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
% This function reads the UTFVP images using 'readUTFVPImages', 
% preprocesses the UTFVP images, does a ROI extraction (finger region 
% localisation) and finger normalisation (rotation compensation)
%
% Parameters:
%  inputPath  - Path to the UTFVP images
%
% Returns:
%  ppImages  - 3D cell array with the preprocessed images
%  roi       - ROI masks corresponding to the images in ppImages
%  roi_small - Smaller ROI masks corresponding to the images in ppImages
% *************************************************************************
function [ppImages, roi, roi_small] = roiExtractPreprocessing(inputPath)
%ROIEXTRACTPREPROCESSING Extracts the ROI from the UTFVP images including
%preprocessing
%   Detailed explanation goes here
    global use_progress_bar;
	% Get the settings
	global settings;
	ls = settings.PreprocessingSettings;	% local settings
	
    % Read the images
    fprintf('Reading input images...\n');
    images = readUTFVPImages(inputPath, '*.png');
    if isempty(images)
        fprintf(2, 'ERROR: Images could not be read. Aborting.\n');
        return;
    end
    [no_users, no_fingers, no_images] = size(images);
    
    % Create empty cell arrays for storage
    ppImages = cell(no_users, no_fingers, no_images);
    roi = cell(no_users, no_fingers, no_images);
    roi_small = cell(no_users, no_fingers, no_images);
    
    if use_progress_bar
        ProgressBar.update('new', 'Preprocessing images ...', 'ROI Extraction and Preprocessing');
        starttime = tic;
        nrTotal = no_users*no_fingers*no_images;
    end
    
    % ROI Extraction and preprocessing
    for u = 1:no_users
        for f = 1:no_fingers
            for i = 1:no_images
                I = images{u, f, i};
                % Resize the image
                img=imresize(I, 0.5);
                img=im2double(img);
                % Do adaptive histogram equalisation
                img=adapthisteq(img);
                % ROI extraction (Lee method)
                [fvr, edges] = lee_region(img, mask_height, mask_width);
                [fvr_small, edges_small]=lee_region_small(img, ls.mask_height, ls.mask_width, ls.add_pix);
                % Normalization (Huang method)
                [~,fvrn_small, ~, ~] = huang_normalise(img, fvr_small, edges_small);
                [Inorm, fvrn, ~, ~] = huang_normalise(img, fvr, edges);
                % Save the images and ROI information
                ppImages{u, f, i} = Inorm;
                roi{u, f, i} = logical(fvrn);
                roi_small{u, f, i} = logical(fvrn_small);
                
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

