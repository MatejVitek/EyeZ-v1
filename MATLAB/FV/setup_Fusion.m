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
% This file is the global setup file for all the necessary parts in the
% fusion framework.
% The most important paths are:
% mainpath ... The main execution path, default is the MATLAB working dir
% utfvp_images_path ... Path where to find the UTFVP images, relative to
%                       mainpath
% iuwt_path ... Path where the ARIA Vessels Lib is located
% masi_path ... Path where the MASI Fusion framework is located
% 
% In addition the parameter "use_progress_bar" can be set to true or false
% if a the progress should be shown using a progress bar or not
%
% This script adds all the path for the utility, feature extraction and
% fusion functions to the MATLAB path and sets up the MASI framework
% *************************************************************************

% MASI Fusion has to be setup first, else other variables are cleared
% MASI Fusion Framework path
masi_path = 'H:\PROGRAMME\Programmierung\Matlab\Image Processing\masi-fusion-scm-2016-04-24';             % Path to the masi fusion framework
% Set-up MASI fusion
addpath(genpath(masi_path));
run(fullfile(masi_path, 'setup_labeling.m'));

% IUWT Vessels Lib path
iuwt_path = 'iuwt_vessels_lib';

% Main execution path
global mainpath;
mainpath = '';

% UTFVP path
global utfvp_images_path; 
utfvp_images_path = 'data/utfvp';   % Put the UTFVP images there

% Add required paths to MATLAB path
addpath('functions_ton');
addpath('functions_feature_extraction');
addpath('functions_fusion');
addpath('functions_eer_evaluation');
addpath('utility_functions');
addpath(genpath(iuwt_path));

% Use progress bars or not
global use_progress_bar
use_progress_bar = true;

% Setup all the settings
setup_Settings;