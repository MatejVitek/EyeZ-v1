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
% This file is the global settings file for storing all different kinds of
% settings, including
% preprocessing
% single feature extraction
% multiple feature extraction
% fusion
% score calculation
%
% A struct variable called settings is provided as global variable which
% contains substructs for each of the above listed processing steps
% *************************************************************************

% Settings struct is a global variable
global settings;

% Sections containing each subsettings
%% General Settings
% Main configuration
override_results = false;           % Override existing results
save_features = true;				% Save the single feature files
save_fused_features = true;         % Save the fused features
do_matching = true;                 % Perform the matching step
save_scores = true;                 % Save the scores after matching
do_eer_calculation = true;          % Perform the EER calculation
save_results = true;                % Save the EER results
matching_mode = 'full';             % Matching mode: train, test, full
free_memory = true;                 % Free memory of unused variables

generalSettings = struct('override_results', override_results, ...
    'save_features', save_features, 'save_fused_features', ...
    save_fused_features, 'do_matching', do_matching, 'save_scores', ...
    save_scores, 'do_eer_calculation', do_eer_calculation, ...
    'save_results', save_results, 'matching_mode', matching_mode, ...
    'free_memory', free_memory);


%% Preprocessing
mask_height = 4;  % Height of the mask used for ROI extraction (lee)
mask_width = 20;  % Width of the mask used for ROI extraction (lee)
add_pix = 6;      % Pixel to add for smaller ROI

preprocessingSettings = struct('mask_height', mask_height, 'mask_width', ...
    mask_width, 'add_pix', add_pix);


%% Feature Extraction (single) - prior to second type of fusion
% Maximum Curvature
mc_sigma = 2.5;
% Repeated Line Tracking
rlt_max_iterations = 3000;
rlt_r = 2;
rlt_W = 17;
% Wide Line Detector
wld_r1 = 7;
wld_t1 = 0.5;
wld_g1 = 64;
% IUWT
iuwt_levels = 2:3;
iuwt_bw_thres = 30;
iuwt_bin_thres = 30;
% Gabor Filter
gb_p0 = [0,0]';     % translation coordinates 
gb_N = 16;          % number of orientations of the filter
gb_bw = 1.5;        % bandwidth
gb_lambda = 8;      % wavelength of cosinusoidal part of the filter (pixel)
gb_gamma = 0.8;     % Aspect ratio (sigma_y*sigma_x) 
gb_sz = 13;         % Size of the kernel of the filter  
% Principal Curvature
pc_sigma = 2;
pc_thresh = 0.5;
pc_bw_thres = 20;

featureExtractionSingleSettings = struct('mc_sigma', mc_sigma, ...
    'rlt_max_iterations', rlt_max_iterations, 'rlt_r', rlt_r, ...
    'rlt_W', rlt_W, 'wld_r1', wld_r1, 'wld_t1', wld_t1, 'wld_g1', ...
    wld_g1, 'iuwt_levels', iuwt_levels, 'iuwt_bw_thres', iuwt_bw_thres, ...
    'iuwt_bin_thres', iuwt_bin_thres, 'gb_p0', gb_p0, 'gb_N', gb_N, ...
    'gb_bw', gb_bw, 'gb_lambda', gb_lambda, 'gb_gamma', gb_gamma, ...
    'gb_sz', gb_sz, 'pc_sigma', pc_sigma, 'pc_thresh', pc_thresh, ...
    'pc_bw_thres', pc_bw_thres);


%% Feature Extraction (multiple, first fusion type)
% Maximum Curvature
sigma=1.9:0.1:2.5;
settingsMC = struct('sigma', sigma);
% IUWT
% Currently done in extract_features_iuwt_multi.m
% Repeated Line Tracking
max_iterations = [3000,4000]; % original value: max_iterations=3000;
r = [1,5];  % original value: r =2
W = [15,17,21]; % original value: W = 17
settingsRLT = struct('max_iterations', max_iterations, 'r', r, 'W', W);
% Wide Line Detector
parameters_matrix=zeros(40,3);
% original value: r=7
r_p=6;
for ii=1:5
    parameters_matrix((ii-1)*8+1:(ii-1)*8+8,1)=r_p;
    r_p=r_p+1;
end
% original value: t=0.5
parameters_matrix(1:2:end,2)=0.5;
parameters_matrix(2:2:end,2)=1;
% original value: g=64
parameters_matrix(1:4,3)=45:2:51;
parameters_matrix(5:8,3)=51:54;
parameters_matrix(9:12,3)=64:2:70;
parameters_matrix(13:16,3)=66:2:72;
parameters_matrix(17:20,3)=86:2:92;
parameters_matrix(21:24,3)=88:2:94;
parameters_matrix(25:28,3)=108:4:120;
parameters_matrix(29:32,3)=118:2:124;
parameters_matrix(33:36,3)=146:2:152;
parameters_matrix(37:40,3)=154:2:160;
settingsWLD = struct('parameters_matrix', parameters_matrix);
% Principal Curvature
sigma=1:0.3:3; % original value: sigma = 2;
thresh=0.3:0.2:1; % original value: thresh = 0.5;
settingsPC = struct('sigma', sigma, 'thresh', thresh);
% Gabor Filter
p0=[0,0]';              % translation coordinates
N = [8,16];             % number of orientations of the filter
%N=16;
%original values: bw=1.5 gamma=0.8; lambda=8; sz=13;
bw = 0.5:0.5:1.5;       % bandwidth (1.5)
lambda=[5,10,14];       % wavelength of cosinusoidal part of the filter (pixel) (10)
gamma = [0.3,0.5,0.9];  % Aspect ratio (sigma_y*sigma_x) (0.9)
%sz=[10,13,15];         % Size of the kernel of the filter  (25)
sz=[10,15];
settingsGB = struct(    'p0', p0, 'N', N, 'bw', bw, 'lambda', lambda, ...
                        'gamma', gamma, 'sz', sz);
                    
featureExtractionMultipleSettings = struct('MC', settingsMC, 'RLT', ...
    settingsRLT, 'WLD', settingsWLD, 'IUWT', [], 'PC', ...
    settingsPC, 'GB', settingsGB);


%% Fusion
% Average (Mean)
% Currently no settings
% Majority Voting
% Currently no settings
% STAPLE
epsilon_S = 0.0001;
init_flag_S = 0;
cf1 = 0;
areaopen_1 = 5;
areaopen_2 = 10;
settingsSTAPLE = struct('epsilon_S', epsilon_S, 'init_flag_S', init_flag_S, ...
                'cf1', cf1, 'areaopen_1', areaopen_1, 'areaopen_2', areaopen_2);
% STAPLER
epsilon_S = 0.0001;
init_flag_S = 0;
cf1 = 0;
areaopen_1 = 5;
areaopen_2 = 10;
settingsSTAPLER = struct('epsilon_S', epsilon_S, 'init_flag_S', init_flag_S, ...
                'cf1', cf1, 'areaopen_1', areaopen_1, 'areaopen_2', areaopen_2);
% COLLATE
epsilon_C = 0.001;
init_flag_C = 0;
prior_flag_C = 0;
%alphas1 = 0;
%alphas2 = 1e15;
alphas3 = [0 1e15];
%alphas4 = [0 1 1e15];
%alphas5 = [0 0.5 2 1e15];
cvals3 = [0.99];
%cvals4 = [0.75 0.99];
%cvals5 = [0.45 0.75 0.99];
areaopen_1 = 5;
areaopen_2 = 10;
settingsCOLLATE = struct('epsilon_C', epsilon_C, 'init_flag_C', init_flag_C, ...
                'prior_flag_C', prior_flag_C, 'alphas3', alphas3, 'cvals3', ...
                cvals3, 'areaopen_1', areaopen_1, 'areaopen_2', areaopen_2);
            
fusionSettings = struct('Av', [], 'MV', [], 'STAPLE', settingsSTAPLE, ...
                'STAPLER', settingsSTAPLER, 'COLLATE', settingsCOLLATE);


%% Score Calculation, Matching
% Configuration of the Miura matcher
cw_default = 80;    % Translation in x direction
ch_default = 30;    % Translation in y direction
% Database specific, number of subjects, fingers, images
no_fing=6;          % Number of fingers per subject
img_per_finger=4;   % Number of images (captures) per finger
no_users = 60;      % Number of subjects

matchingSettings = struct('cw_default', cw_default, 'ch_default', ...
    ch_default, 'no_fing', no_fing, 'img_per_finger', img_per_finger, ...
    'no_users', no_users);


%% Construct the settings object
settings = struct(  'PreprocessingSettings', preprocessingSettings, ...
                    'FeatureExtractionSingleSettings', featureExtractionSingleSettings, ...
                    'FeatureExtractionMultipleSettings', featureExtractionMultipleSettings, ...
                    'FusionSettings', fusionSettings, ...
                    'MatchingSettings', matchingSettings, ...
                    'GeneralSettings', generalSettings);