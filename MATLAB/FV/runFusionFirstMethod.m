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
% This function runs the first fusion method (only a single feature type) 
% for one of the six features and the given fusion strategy including the
% matching and EER calculation step
%
% Parameters:
%  featureType       - The feature extractor type, can either be
%                      'MC', 'PC', 'GB', 'WLD', 'IUWT' or 'RLT'
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
function EERs = runFusionFirstMethod(featureType, fusion_strategy, do_setupFusion)
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
    global mainpath;
    global utfvp_images_path; 
    global free_memory;
    free_memory = ls.free_memory;   % Compatibility
    
    % Ask user which feature extraction method
    if nargin < 1
        featureTypes = {'MC', 'PC', 'GB', 'WLD', 'IUWT', 'RLT'};
        % Construct a menu dialog and let the user choose
        featureTypeUI = menu('Feature Extraction Type', featureTypes{:});
        if featureTypeUI == 0
            fprintf(2, 'ERROR: No valid feature extraction type provided.\n');
            return;
        end
        featureType = featureTypes{featureTypeUI};
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

    
    %% Check if db and roi files are present
    fprintf('Checking if DB and ROI present ...\n');
    if ls.override_results || ~exist(fullfile(mainpath, 'data', 'db.mat'), 'file') || ...
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
        load(fullfile(mainpath, 'data', 'db.mat'));
        load(fullfile(mainpath, 'data', 'roi.mat'));
        load(fullfile(mainpath, 'data', 'roi_small.mat'));
        fprintf('DB and ROI found and loaded.\n');
    end
    
    %% Do the feature extraction
    fprintf('Run feature extraction (%s)...\n', featureType);
    % Check if the features already exist and if yes simply load
    % the feature file if override_results is set to false
    if ls.override_results || ~exist(fullfile(mainpath, 'data', 'features_single_fused', ['feat_' featureType '_multi.mat']), 'file')
        switch featureType
            case 'MC'   % Maximum Curvature
                features_multi = extract_features_mc_multi(db, roi, roi_small);
            case 'WLD'  % Wide Line Detector
                features_multi = extract_features_wld_multi(db, roi, roi_small);
            case 'GB'   % Gabor Filter
                features_multi = extract_features_gb_multi(db, roi, roi_small);
            case 'IUWT' % 
                features_multi = extract_features_iuwt_multi(db, roi, roi_small);
            case 'PC'   % Principal Curvature
                features_multi = extract_features_pc_multi(db, roi, roi_small);
            case 'RLT'  % Reapeated Line Tracking
                features_multi = extract_features_rlt_multi(db, roi, roi_small);
        end
        
        % Save the multi extracted feature file
        if ls.save_features
            fprintf('Saving features file ...\n');
            if ~exist(fullfile(mainpath, 'data', 'features_single_fused'), 'dir')   % Create dir if it does not exist
                mkdir(fullfile(mainpath, 'data', 'features_single_fused'));
            end
            save(fullfile(mainpath, 'data', 'features_single_fused', ['feat_' featureType '_multi.mat']), 'features_multi', '-v7.3');
        end
    else
        % Load the features
        fprintf('Feature file already present. Load feature file...\n');
        load(fullfile(mainpath, 'data', 'features_single_fused', ['feat_' featureType '_multi.mat']));
    end
    
    if ls.free_memory
        clear db;   % db is no longer needed
    end
    
    
    %% Do the feature fusion
    fprintf('Run feature fusion ...\n');
    if ls.override_results || ~exist(fullfile(mainpath, 'data', 'features_single_fused', ['feat_' featureType '_' fusion_strategy '.mat']), 'file')
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
                fused_features = fuse_features_COLLATE(features_multi, roi_small, true);
        end
        
        % Save the fused feature file
        if ls.save_fused_features
            fprintf('Saving fused features file ...\n');
            if ~exist(fullfile(mainpath, 'data', 'features_single_fused'), 'dir')   % Create dir if it does not exist
                mkdir(fullfile(mainpath, 'data', 'features_single_fused'));
            end
            save(fullfile(mainpath, 'data', 'features_single_fused', ['feat_' featureType '_' fusion_strategy '.mat']), 'fused_features', '-v7.3');
        end
    else
        % Load the fused features
        fprintf('Fused feature file already present. Load feature file...\n');
        load(fullfile(mainpath, 'data', 'features_single_fused', ['feat_' featureType '_' fusion_strategy '.mat']));
    end
    
    if ls.free_memory
        clear features_multi;   % features_multi is no longer needed
    end
    
    % Create the struct with all fused features
%     fused_features.('MV') = features_mv;
%     fused_features.('Av') = features_av;
%     fused_features.('STAPLE') = features_staple;
%     fused_features.('STAPLE_prob') = features_staple_prob;
%     fused_features.('STAPLER') = features_stapler;
%     fused_features.('STAPLER_prob') = features_stapler_prob;
%     fused_features.('COLLATE') = features_collate;
%     fused_features.('COLLATE_prob') = features_collate_prob;
%     % Save the features
%     if save_fused_features
%         save(fullfile(mainpath, 'fused_features_type1', ['features_' featureType '.mat']), ...
%             'features_mv', 'features_av', 'features_staple', 'features_staple_prob', ...
%             'features_stapler', 'features_stapler_prob', 'features_collate', ...
%             'features_collate_prob');
%     end

    %% Do the matching
    if ls.do_matching
        fprintf('Run matching ...\n');
        if ls.override_results || ~exist(fullfile(mainpath, 'data', 'scores_fusion_1', ['scores_' featureType '_' fusion_strategy '_' ls.matching_mode '.mat']), 'file')
%        scores = struct();
%         % Do the matching for each fusion variant
%         for i=1:numel(fusion_strategies)
%             [gen_scores, imp_scores] = computeScoresFull(fused_features.(fusion_strategies{i}));
%             scores.(fusion_strategies{i}) = struct('fusion_type', fusion_strategies{i}, 'genuine_scores', gen_scores, 'impostor_scores', imp_scores);
%         end
            [gen_scores, imp_scores] = computeScoresFull(fused_features, 'MatchMode', ls.matching_mode);
            scores = struct('fusion_strategy', fusion_strategy, 'featureType', ...
                featureType, 'match_mode', ls.matching_mode, ...
                'genuine_scores', gen_scores, 'impostor_scores', imp_scores);
            if ls.save_scores
                fprintf('Saving fusion scores file ...\n');
                if ~exist(fullfile(mainpath, 'data', 'scores_fusion_1'), 'dir')   % Create dir if it does not exist
                    mkdir(fullfile(mainpath, 'data', 'scores_fusion_1'));
                end
               save(fullfile(mainpath, 'data', 'scores_fusion_1', ['scores_' featureType '_' fusion_strategy '_' ls.matching_mode '.mat']), 'scores', '-v7.3');
            end
        else
            % Load the scores
            fprintf('Scores file already present. Load scores file...\n');
            load(fullfile(mainpath, 'data', 'scores_fusion_1', ['scores_' featureType '_' fusion_strategy '_' ls.matching_mode '.mat']));
        end
    end

    %% Determine EER
    if ls.do_eer_calculation
        fprintf('Determine EER and ROC/DET curves ...\n');
        if ls.override_results || ~exist(fullfile(mainpath, 'data', 'results_fusion_1', ['eers_' featureType '_' fusion_strategy '_' ls.matching_mode '.mat']), 'file')
            EERs = struct('feature_type', featureType, 'fusion_strategy', fusion_strategy, 'match_mode', ls.matching_mode);
            plots = struct('feature_type', featureType, 'fusion_strategy', fusion_strategy, 'match_mode', ls.matching_mode);
    %         for i=1:numel(fusion_strategies)
    %             current_scores = scores.(fusion_strategies{i});
    %             [EERs.(fusion_strategies{i}), ~, ~, ~, plots.(fusion_strategies{i})] = EER_DET_conf(current_scores.genuine_scores, current_scores.impostor_score, 1, 10000);
    %             fprintf('Resulting EER for %s: %7.6f%%\n', fusion_strategies{i}, EERs.(fusion_strategies{i}));
    %         end
            [EERs.eer, ~, ~, ~, plots.plot] = EER_DET_conf(scores.genuine_scores, scores.impostor_scores, 1, 10000);
            % Save the results
            if ls.save_results
                fprintf('Saving EER results file ...\n');
                if ~exist(fullfile(mainpath, 'data', 'results_fusion_1'), 'dir')   % Create dir if it does not exist
                    mkdir(fullfile(mainpath, 'data', 'results_fusion_1'));
                end
                save(fullfile(mainpath, 'data', 'results_fusion_1', ['eers_' featureType '_' fusion_strategy '_' ls.matching_mode '.mat']), 'EERs', 'plots', '-v7.3');
            end
        else
            % Load the results file
            fprintf('Results file already present. Load results file...\n');
            load(fullfile(mainpath, 'data', 'results_fusion_1', ['eers_' featureType '_' fusion_strategy '_' ls.matching_mode '.mat']));
        end
        fprintf('Resulting EER for %s, %s: %7.6f%%\n', featureType, fusion_strategy, EERs.eer);
    end
end