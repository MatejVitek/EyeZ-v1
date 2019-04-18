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
% This function runs the second fusion method (several feature types) 
% for all SIX features (or a subset), including the matching and EER 
% calculation step
%
% Parameters:
%  featuresSelected  - The selected feature extractors according to Tab. 3
%                      in the paper, options are:
%                       1... MC + RLT + PC + WLD
%                       2... MC + PC + WLD + GF
%                       3... MC + PC + WLD + IUWT
%                       4... MC + RLT + PC + WLD + GF
%                       5... MC + RLT + PC + WLD + IUWT
%                       6... MC + RLT + PC + WLD + IUWT + GF
%                       7... 2*MC + RLT + PC + WLD + IUWT + GF
%  fusion_strategy   - The fusion strategy to use, can either be
%                      'MV', 'Av', 'STAPLE', 'STAPLE_prob', 'STAPLER', 
%           		   'STAPLER_prob', 'COLLATE' or 'COLLATE_prob'
%  do_setupFusion    - Perfom fusion setup (if function is called multiple
%                      times this should only be done once as the MASI
%                      setup clears all variables
% If no parameters are given, the function asks the user to provide the
% feature type and the fusion scheme (choice menu)
%
% Returns:
%  EERs - A struct containing the resulting EER (%) and the ROC plot data
% *************************************************************************
function EERs = runFusionSecondMethod(featuresSelected, fusion_strategy, do_setupFusion)
    %% Prequisites
    if nargin < 3
        do_setupFusion = true;  % default value
    end
    if do_setupFusion
        setup_Fusion;
    end
    EERs = [];

    %% Configuration
    % Get the settings
	global settings;
	ls = settings.GeneralSettings;	% local settings
    global free_memory;
    free_memory = ls.free_memory;   % Compatibility
  
    % Ask user which features to combine
    % According to the paper (table 3) there are 7 different combinations:
    % 1... MC + RLT + PC + WLD
    % 2... MC + PC + WLD + GF
    % 3... MC + PC + WLD + IUWT
    % 4... MC + RLT + PC + WLD + GF
    % 5... MC + RLT + PC + WLD + IUWT
    % 6... MC + RLT + PC + WLD + IUWT + GF
    % 7... 2*MC + RLT + PC + WLD + IUWT + GF
    if nargin < 1
        featureTypes = {'MC, RLT, PC, WLD', ...
                        'MC, PC, WLD, GF', ...
                        'MC, PC, WLD, IUWT', ...
                        'MC, RLT, PC, WLD, GF', ...
                        'MC, RLT, PC, WLD, IUWT', ...
                        'MC, RLT, PC, WLD, IUWT, GF' ...
                        '2*MC, RLT, PC, WLD, IUWT, GF'};
        % Construct a menu dialog and let the user choose
        featureTypeUI = menu('Feature Extraction Type', featureTypes{:});
        if featureTypeUI == 0
            fprintf(2, 'ERROR: No valid feature extraction type provided.\n');
            return;
        end
        featuresSelected = featureTypeUI;
    end
	
	% Ask user which fusion strategy
    if nargin < 2
        fusion_strategies = {'MV', 'Av', 'STAPLE', 'STAPLE_prob', 'STAPLER', ...
			'STAPLER_prob', 'COLLATE', 'COLLATE_prob'};
        % Construct a menu dialog and let the user choose
        fusionStrategyUI = menu('Fusion Strategy', fusion_strategies{:});
        if fusionStrategyUI == 0
            fprintf(2, 'ERROR: No valid fusion strategy provided.\n');
            return;
        end
        fusion_strategy = fusion_strategies{fusionStrategyUI};
    end
    
    
    %% Main process
    % Determine EER
    if ls.do_eer_calculation
        EERs = calculateLoadEER(ls.override_results, ls.save_features, featuresSelected, fusion_strategy, ls.save_fused_features, ls.matching_mode, ls.save_scores, ls.save_results);
    elseif do_matching
        % Do the matching
        scores = calculateLoadScores(ls.override_results, ls.save_features, featuresSelected, fusion_strategy, ls.save_fused_features, ls.matching_mode, ls.save_scores);
    else
        % if EER calculation and score calculation is not chosen do at
        % least the fusion and if necessary feature extraction and
        % preprocessing
        % Run the fusion
        fused_features = fuseLoadFeatures(ls.override_results, ls.save_features, featuresSelected, fusion_strategy, ls.save_fused_features);
    end
end

%% Main part functions - Preprocessing, Feature Extraction, Fusion, Score Calculation, EER Calculation
% Calculate or load the EER and ROC plots
function EERs = calculateLoadEER(override_results, save_features, featuresSelected, fusion_strategy, save_fused_features, matching_mode, save_scores, save_results)
    global mainpath;
    
    % Initialise all output variables
    EERs = [];

    if override_results || ~exist(fullfile(mainpath, 'data', 'results_fusion_2', ['eers_' num2str(featuresSelected) '_' fusion_strategy '_' matching_mode '.mat']), 'file')
        EERs = struct('feature_selected', num2str(featuresSelected), 'fusion_strategy', fusion_strategy, 'match_mode', matching_mode);
        plots = struct('feature_selected', num2str(featuresSelected), 'fusion_strategy', fusion_strategy, 'match_mode', matching_mode);
        % Load or calculate scores prior to EER calculation
        scores = calculateLoadScores(override_results, save_features, featuresSelected, fusion_strategy, save_fused_features, matching_mode, save_scores);
        % Do the EER calculation
        fprintf('Determine EER and ROC/DET curves ...\n');
        [EERs.eer, ~, ~, ~, plots.plot] = EER_DET_conf(scores.genuine_scores, scores.impostor_scores, 1, 10000);
        % Save the results
        if save_results
            fprintf('Saving EER results file ...\n');
            if ~exist(fullfile(mainpath, 'data', 'results_fusion_2'), 'dir')   % Create dir if it does not exist
                mkdir(fullfile(mainpath, 'data', 'results_fusion_2'));
            end
            save(fullfile(mainpath, 'data', 'results_fusion_2', ['eers_' num2str(featuresSelected) '_' fusion_strategy '_' matching_mode '.mat']), 'EERs', 'plots', '-v7.3');
        end
    else
        % Load the results file
        fprintf('Results EER file already present. Load results file...\n');
        load(fullfile(mainpath, 'data', 'results_fusion_2', ['eers_' num2str(featuresSelected) '_' fusion_strategy '_' matching_mode '.mat']));
    end
    fprintf('Resulting EER for %s, %s, %s set: %7.6f%%\n', num2str(featuresSelected), fusion_strategy, matching_mode, EERs.eer);
end

% Calculate or load the scores
function scores = calculateLoadScores(override_results, save_features, featuresSelected, fusion_strategy, save_fused_features, matching_mode, save_scores)
    global mainpath;
    
    % Initialise all output variables
    scores = [];

    if override_results || ~exist(fullfile(mainpath, 'data', 'scores_fusion_2', ['scores_' num2str(featuresSelected) '_' fusion_strategy '_' matching_mode '.mat']), 'file')
        % Load or fuse the features prior to score calculation
        fused_features = fuseLoadFeatures(override_results, save_features, featuresSelected, fusion_strategy, save_fused_features);
        % Do the score calculation
        fprintf('Run matching ...\n');
        [gen_scores, imp_scores] = computeScoresFull(fused_features, 'MatchMode', matching_mode);
        scores = struct('fusion_strategy', fusion_strategy, 'featuresSelected', ...
            'match_mode', matching_mode, num2str(featuresSelected), ...
            'genuine_scores', gen_scores, 'impostor_scores', imp_scores);
        if save_scores
            fprintf('Saving scores file ...\n');
            if ~exist(fullfile(mainpath, 'data', 'scores_fusion_2'), 'dir')   % Create dir if it does not exist
                mkdir(fullfile(mainpath, 'data', 'scores_fusion_2'));
            end
           save(fullfile(mainpath, 'data', 'scores_fusion_2', ['scores_' num2str(featuresSelected) '_' fusion_strategy '_' matching_mode '.mat']), 'scores', '-v7.3');
        end
    else
        % Load the scores
        fprintf('Scores file already present. Load scores file...\n');
        load(fullfile(mainpath, 'data', 'scores_fusion_2', ['scores_' num2str(featuresSelected) '_' fusion_strategy '_' matching_mode '.mat']));
    end
end

% Fuse the features or load fused features
function fused_features = fuseLoadFeatures(override_results, save_features, featuresSelected, fusion_strategy, save_fused_features)
    global free_memory;
    global mainpath;
    
    % Initialise all output variables
    fused_features = [];

    if override_results || ~exist(fullfile(mainpath, 'data', 'features_multi_fused', ['feat_' num2str(featuresSelected) '_' fusion_strategy '.mat']), 'file')
        % Load or extract the single features prior to fusion
        [features_mc, features_gb, features_wld, features_pc, features_rlt, features_iuwt, roi, roi_small] = extractLoadFeatures(override_results, save_features); 
        % Run the feature fusion
        fprintf('Run feature fusion ...\n');
        % Construct the features array     
        switch featuresSelected
            case 1
                % 1... MC + RLT + PC + WLD
                features_multi = cellfun(@combine_features, features_mc, features_rlt, features_pc, features_wld, 'UniformOutput', false);
            case 2
                % 2... MC + PC + WLD + GF
                features_multi = cellfun(@combine_features, features_mc, features_pc, features_wld, features_gb, 'UniformOutput', false);
            case 3
                % 3... MC + PC + WLD + IUWT
                features_multi = cellfun(@combine_features, features_mc, features_pv, features_wld, features_iuwt, 'UniformOutput', false);
            case 4
                % 4... MC + RLT + PC + WLD + GF
                features_multi = cellfun(@combine_features, features_mc, features_rlt, features_pc, features_wld, features_gb, 'UniformOutput', false);
            case 5
                % 5... MC + RLT + PC + WLD + IUWT
                features_multi = cellfun(@combine_features, features_mc, features_rlt, features_pc, features_wld, features_iuwt, 'UniformOutput', false);
            case 6
                % 6... MC + RLT + PC + WLD + IUWT + GF
                features_multi = cellfun(@combine_features, features_mc, features_rlt, features_pc, features_wld, features_iuwt, features_gb, 'UniformOutput', false);
            case 7
                % 7... 2*MC + RLT + PC + WLD + IUWT + GF
                features_multi = cellfun(@combine_features, features_mc, features_rlt, features_pc, features_wld, features_iuwt, features_gb, 'UniformOutput', false);
        end
        % Free memory of the individual features after they have been 
        % combined into a single cell array
        if free_memory
            clear features_mc features_pc features_wld features_rlt features_gb features_iuwt;   % single features are no longer needed
        end
        % Do the fusion
        switch fusion_strategy
            case 'MV'
                fused_features = fuse_features_MV(features_multi);
            case 'Av'
                fused_features = fuse_features_Av(features_multi);
            case 'STAPLE'
                fused_features = fuse_features_STAPLE(features_multi, roi_small, false);
            case 'STAPLER'
                fused_features = fuse_features_STAPLER(features_multi, roi_small, false);
            case 'COLLATE'
                fused_features = fuse_features_COLLATE(features_multi, roi_small, false);
            case 'STAPLE_prob'
                fused_features = fuse_features_STAPLE(features_multi, roi_small, true);
            case 'STAPLER_prob'
                fused_features = fuse_features_STAPLER(features_multi, roi_small, true);
            case 'COLLATE_prob'
                if featuresSelected == 6    % All 6 features, equal weights
                    % Special case for best result of EER 0.1853% which is
                    % obtained using custom fusion parameters listed below
                    probValues = true;
                    doBW = false;
                    doMorph = false;
                    doBinarize = false;
                    collateEpsilon = 0.1;
                    collatePriorFlag = 0;
                    collateInitFlag = 0;
                    collateAlphas = [1, 1, 1];
                    collateCVals = [0.99, 0.95];
                    fused_features = FeatureFusion(features_multi, 'Collate', ...
                        'SaveProb', probValues, 'DoBW', doBW, 'DoMorphological', ...
                        doMorph, 'CollateEpsilon', collateEpsilon, 'DoBinarize', ...
                        doBinarize, 'CollateInit_flag', collateInitFlag, ...
                        'CollatePrior_flag', collatePriorFlag, ...
                        'CollateAlphas', collateAlphas, 'CollateCvals', collateCVals);
                else
                    fused_features = fuse_features_COLLATE(features_multi, roi_small, true);
                end
        end
        
        % Save the fused feature file
        if save_fused_features
            fprintf('Saving fused features file ...\n');
            if ~exist(fullfile(mainpath, 'data', 'features_multi_fused'), 'dir')   % Create dir if it does not exist
                mkdir(fullfile(mainpath, 'data', 'features_multi_fused'));
            end
            save(fullfile(mainpath, 'data', 'features_multi_fused', ['feat_' num2str(featuresSelected) '_' fusion_strategy '.mat']), 'fused_features', '-v7.3');
        end
    else
        % Load the fused features
        fprintf('Fused feature file already present. Load feature file...\n');
        load(fullfile(mainpath, 'data', 'features_multi_fused', ['feat_' num2str(featuresSelected) '_' fusion_strategy '.mat']));
        
        if free_memory
            clear features_mc features_pc features_wld features_rlt features_gb features_iuwt;   % single features are no longer needed
        end
    end
end

% Extract the single features or load extracted features
function [features_mc, features_gb, features_wld, features_pc, features_rlt, features_iuwt, roi, roi_small] = extractLoadFeatures(override_results, save_features)
    global free_memory;
    global mainpath;
    
    % Initialise all output variables
    features_mc = [];
    features_gb = [];
    features_wld = [];
    features_rlt = [];
    features_iuwt = [];
    features_pc = [];
	
    fprintf('Checking if all feature files present ...\n');
    if override_results || ~exist(fullfile(mainpath, 'data', 'features_single', 'features_mc.mat'), 'file') || ...
            ~exist(fullfile(mainpath, 'data', 'features_single', 'features_gb.mat'), 'file') || ...
            ~exist(fullfile(mainpath, 'data', 'features_single', 'features_pc.mat'), 'file') || ...
            ~exist(fullfile(mainpath, 'data', 'features_single', 'features_wld.mat'), 'file') || ...
            ~exist(fullfile(mainpath, 'data', 'features_single', 'features_rlt.mat'), 'file') || ...
            ~exist(fullfile(mainpath, 'data', 'features_single', 'features_iuwt.mat'), 'file')
        % Load the database and ROI files prior to feature extraction
        [db, roi, roi_small] = preprocessLoadImageSet(override_results);
        % Extract the features
        fprintf('One of the features files not found. Running feature extraction ...\n');
        [features_mc, features_gb, features_wld, features_pc, features_rlt, features_iuwt] = extractSingleFeatures(db, roi, roi_small);
        % Save the feature files
        if save_features
            fprintf('Saving feature files ...\n');
            if ~exist(fullfile(mainpath, 'data', 'features_single'), 'dir')   % Create dir if it does not exist
                mkdir(fullfile(mainpath, 'data', 'features_single'));
            end
            save(fullfile(mainpath, 'data', 'features_single', 'features_mc.mat'), 'features_mc', '-v7.3');
            save(fullfile(mainpath, 'data', 'features_single', 'features_gb.mat'), 'features_gb', '-v7.3');
            save(fullfile(mainpath, 'data', 'features_single', 'features_wld.mat'), 'features_wld', '-v7.3');
            save(fullfile(mainpath, 'data', 'features_single', 'features_pc.mat'), 'features_pc', '-v7.3');
            save(fullfile(mainpath, 'data', 'features_single', 'features_rlt.mat'), 'features_rlt', '-v7.3');
            save(fullfile(mainpath, 'data', 'features_single', 'features_iuwt.mat'), 'features_iuwt', '-v7.3');
        end
    else
        fprintf('Feature files already present. Load features file ...\n');
        load(fullfile(mainpath, 'data', 'features_single', 'features_mc.mat'));
        load(fullfile(mainpath, 'data', 'features_single', 'features_gb.mat'));
        load(fullfile(mainpath, 'data', 'features_single', 'features_wld.mat'));
        load(fullfile(mainpath, 'data', 'features_single', 'features_pc.mat'));
        load(fullfile(mainpath, 'data', 'features_single', 'features_rlt.mat'));
        load(fullfile(mainpath, 'data', 'features_single', 'features_iuwt.mat'));
        % Load the ROI files, assume that they are also already/still there
        [~, roi, roi_small] = preprocessLoadImageSet(override_results, true);
        fprintf('Feature and ROI files present and loaded.\n');
    end
    
    if free_memory
        clear db;   % db is no longer needed
    end
end

% Preprocess the images or load images and ROI information
function [db, roi, roi_small] = preprocessLoadImageSet(override_results, roi_only)
    global mainpath;
    global utfvp_images_path; 
    
    % Initialise all output variables
    db = [];
    roi = [];
    roi_small = [];
    
    fprintf('Checking if DB and ROI present ...\n');
    if override_results || ~exist(fullfile(mainpath, 'data', 'db.mat'), 'file') || ...
            ~exist(fullfile(mainpath, 'data', 'roi.mat'), 'file') || ...
            ~exist(fullfile(mainpath, 'data', 'roi_small.mat'), 'file')
        % Create the DB and ROI files
        fprintf('DB file not found. Creating db file ...\n');
        [db, roi, roi_small] = roiExtractPreprocessing(utfvp_images_path);
        % Save the db and roi files
        fprintf('Saving DB and ROI files ...\n');
        if ~exist(fullfile(mainpath, 'data'), 'dir')   % Create dir if it does not exist
            mkdir(fullfile(mainpath, 'data'));
        end
        save(fullfile(mainpath, 'data', 'db.mat'), 'db', '-v7.3');
        save(fullfile(mainpath, 'data', 'roi.mat'), 'roi', '-v7.3');
        save(fullfile(mainpath, 'data', 'roi_small.mat'), 'roi_small', '-v7.3');
    else
        % Load only ROI files and not the DB itself (if already present)
        if ~roi_only
            load(fullfile(mainpath, 'data', 'db.mat'));
        end
        load(fullfile(mainpath, 'data', 'roi.mat'));
        load(fullfile(mainpath, 'data', 'roi_small.mat'));
        fprintf('DB and ROI found and loaded.\n');
    end
end


%% Helper functions
function combinedFeatures = combine_features(varargin)
% Combine the input features into a single 3D feature file
    combinedFeatures = zeros(size(varargin{1}, 1), size(varargin{1}, 2), length(varargin));
    for i=1:length(varargin)
        combinedFeatures(:,:,i) = varargin{i};
    end
end