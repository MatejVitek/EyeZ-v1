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
% This function provides a wrapper for all the different fusion schemes,
% including a bunch of additional parameters that can be set using varargin
%
% Parameters:
%  inputFeatureFiles    - Input feature files as 3D cell array (60x6x4)
%  fusionMethod         - The selected fusion method
%  varargin - Optional additional parameters for the fusion
%
% Returns:
%  fusedFeatures - 3D cell array containing the fused features
% *************************************************************************
function fusedFeatures = FeatureFusion(inputFeatureFiles, fusionMethod, varargin)
%FEATUREFUSION Fuse the given features according to fusion method
%   Fuse all the given features according to the given fusion method and
%   its parameters. Default parameters are used if no parameters are given.

global use_progress_bar;
    
%% Default parameter values
defaultFusionMethod = 'MajorityVote';
defaultWeights = [];    % empty weights, just use the same weights
defaultDoBinarize = true;
defaultBinarizationMethod = 'Median';
defaultDoBW = true;
defaultDoMorphological = false;
defaultBWOperation = 'skel';
defaultBWTimes = 5;
defaultBinThreshold = 0;    % If 0 use median as threshold
defaultBWFirstPar = 30;     % 30 for staple, 50 for the others
defaultBWSecondPar = 15;    % 15 for staple
defaultSaveProb = false;
defaultParallelProcessing = false;
% Parameters STAPLE
defaultStapleEpsilon = 0.0001;
defaultStapleInit_flag = 0;
defaultStaplePrior_flag = -1;
defaultStapleCf = 1;
% Fusion Parameters COLLATE
defaultCollateEpsilon = 0.001;
defaultCollateInit_flag = 0;
defaultCollatePrior_flag = 0;
defaultCollateAlphas = [0 1e15];
defaultCollateCvals = [0.99];
% Fusion Parameters SIMPLE
defaultSimpleNumIterKeep = 3;
defaultSimpleAlpha = 2;
defaultSimplePerfType = 0;      %0, 1 or 2
% Fusion Parameres LogOdds
defaultLogOddsDecayCoeff = 1;

%% Configuration
if nargin < 2
    fusionMethods = {'WeightedMean', 'WeightedSum', 'MajorityVote', ...
        'Staple', 'Stapler', 'Collate', 'SpatialStaple'};
    % Construct a menu dialog with feature type selection
    fusionMethodUI = menu('Feature Type', fusionMethods{:});
    if fusionMethodUI == 0
        fprintf(2, 'WARNING: No feature type selected. Using default feature type: %s\n', defaultFusionMethod);
        %return;
        fusionMethod = defaultFusionMethod;
    else
        fusionMethod = fusionMethods{fusionMethodUI};
    end
end

%% Get additional options using inputparser
p = inputParser;
p.KeepUnmatched = true;
addParamValue(p, 'Weights', defaultWeights, @isnumeric);
addParamValue(p, 'DoBinarize', defaultDoBinarize, @islogical);
addParamValue(p, 'BinarizationMethod', defaultBinarizationMethod, @ischar);
addParamValue(p, 'DoBW', defaultDoBW, @islogical);
addParamValue(p, 'DoMorphological', defaultDoMorphological, @islogical);
addParamValue(p, 'BWTimes', defaultBWTimes, @isscalar);
addParamValue(p, 'BWOperation', defaultBWOperation, @ischar);
addParamValue(p, 'BinThreshold', defaultBinThreshold, @isscalar);
addParamValue(p, 'BWFirstPar', defaultBWFirstPar, @isscalar);
addParamValue(p, 'BWSecondPar', defaultBWSecondPar, @isscalar);
addParamValue(p, 'SaveProb', defaultSaveProb, @islogical);
addParamValue(p, 'ParallelProcessing', defaultParallelProcessing, @islogical);
% Staple
addParamValue(p, 'StapleEpsilon', defaultStapleEpsilon, @isscalar);
addParamValue(p, 'StapleInit_flag', defaultStapleInit_flag, @isscalar);
addParamValue(p, 'StaplePrior_flag', defaultStaplePrior_flag, @isscalar);
addParamValue(p, 'StapleCf', defaultStapleCf, @isscalar);
% Collate
addParamValue(p, 'CollateEpsilon', defaultCollateEpsilon, @isscalar);
addParamValue(p, 'CollateInit_flag', defaultCollateInit_flag, @isscalar);
addParamValue(p, 'CollatePrior_flag', defaultCollatePrior_flag, @isscalar);
addParamValue(p, 'CollateAlphas', defaultCollateAlphas, @isnumeric);
addParamValue(p, 'CollateCvals', defaultCollateCvals, @isnumeric);
% Simple
addParamValue(p, 'SimpleNumIterKeep', defaultSimpleNumIterKeep, @isscalar);
addParamValue(p, 'SimpleAlpha', defaultSimpleAlpha, @isscalar);
addParamValue(p, 'SimplePerfType', defaultSimplePerfType, @isscalar);
% LogOdds
addParamValue(p, 'LogOddsDecayCoeff', defaultLogOddsDecayCoeff, @isscalar);



parse(p, varargin{:});
params = p.Results;

weights = params.Weights;
parallelProcessing = params.ParallelProcessing;
bWFirstPar = params.BWFirstPar;
bWSecondPar = params.BWSecondPar;
doBinarize = params.DoBinarize;
binarizationMethod = params.BinarizationMethod;
binThreshold = params.BinThreshold;
doBW = params.DoBW;
doMorphological = params.DoMorphological;
bwOperation = params.BWOperation;
bwTimes = params.BWTimes;
saveProb = params.SaveProb;
% Staple
stapleEpsilon = params.StapleEpsilon;
stapleInit_flag = params.StapleInit_flag;
staplePrior_flag = params.StaplePrior_flag;
stapleCf = params.StapleCf;
% Collate
collateEpsilon = params.CollateEpsilon;
collateInit_flag = params.CollateInit_flag;
collatePrior_flag = params.CollatePrior_flag;
collateAlphas = params.CollateAlphas;
collateCvals = params.CollateCvals;
% Simple
simpleNumIterKeep = params.SimpleNumIterKeep;
simpleAlpha = params.SimpleAlpha;
simplePerfType = params.SimplePerfType;
% LogOdds
logOddsDecayCoeff = params.LogOddsDecayCoeff;


%% Fusion process

% Load all input features
fprintf('Reading/adapting the input features...\n');
if isempty(inputFeatureFiles)
    fprintf(2, 'ERROR: No input feature paths given.\n');
    return;
end
if numel(size(inputFeatureFiles)) == 3
    % inputFeatureFiles is no string array containing filenames but the
    % feature cell array itself (60x6x4)
    [nr_users, nr_fingers, nr_images] = size(inputFeatureFiles);
    inputFeaturesCell = reshape(inputFeatureFiles, size(inputFeatureFiles, 1)*size(inputFeatureFiles, 2)*size(inputFeatureFiles, 3), []);
    % adapt the cell array to be a 4D vector (img_h, img_w, nr_features,
    % nr_images)
    img_h = size(inputFeaturesCell{1}, 1);
    img_w = size(inputFeaturesCell{1}, 2);
    nrFeatures = size(inputFeaturesCell{1}, 3);
    inputFeatures = false(img_h, img_w, nrFeatures, length(inputFeaturesCell));
    for i=1:length(inputFeaturesCell)
        inputFeatures(:,:,:,i) = inputFeaturesCell{i};
    end
    clear inputFeatureFiles;        % free memory
else
    for i=1:length(inputFeatureFiles)
        featureSet = load(inputFeatureFiles{i});
        % Get the features, no matter which name
        fields = fieldnames(featureSet);
        featureSet = featureSet.(fields{1});
        % Get the dimensions
        [nr_users, nr_fingers, nr_images] = size(featureSet);
        features = reshape(featureSet, nr_users*nr_fingers*nr_images, []);
        % Get the image size and number of images
        % Assumption that all feature files have same number of images and sizes
        if i == 1
            img_h = size(features{1}, 1);
            img_w = size(features{1}, 2);
            nrFeatures = size(features, 1);
            inputFeatures = false(img_h, img_w, nrFeatures, length(inputFeatureFiles));
        end
        inputFeatures(:,:,i,:) = reshape([features{:}], img_h, img_w, []);
    end
end

% Set the missing parameters
%% Enable parallel pool
if parallelProcessing
    poolobj = gcp('nocreate'); % If no pool, do not create new one.
    if isempty(poolobj)
        poolobj = parpool;  % Create default parallel pool
    end
    maxWorkers = Inf;
else
    maxWorkers = 0;         % Run the parfor as for loop
end
if isempty(weights)
%     weights = 1/length(inputFeatureFiles) * ones(length(inputFeatureFiles), 1);
    weights = ones(length(inputFeatures), 3);
end


% Fuse according to given fusion method
fprintf('Fusing features using %s...\n', fusionMethod);
starttime = tic;
nrImages = nr_users*nr_fingers*nr_images;
if use_progress_bar
    if parallelProcessing
        parfor_progress(nrImages, starttime);
    else
        ProgressBar.update('new', 'Fusing Features', [fusionMethod ' Feature Fusion']);
    end
end

fusedFeatures = cell(size(inputFeatures(1),3), 1);

% parfor(i=1:nrImages, maxWorkers)
for i=1:nrImages
    currentFeatures = inputFeatures(:,:,:,i);
    switch fusionMethod
        case 'WeightedMean'
            % Apply the weights, TODO: without a loop
            for f=1:size(currentFeatures,3)
                currentFeatures(:,:,f) = currentFeatures(:,:,f)*weights(f);
            end
%             feat_fused = sum(currentFeatures, 3);
            feat_fused = mean(currentFeatures, 3);
        case 'WeightedSum'
            % Apply the weights, TODO: without a loop
            for f=1:size(currentFeatures,3)
                currentFeatures(:,:,f) = currentFeatures(:,:,f)*weights(f);
            end
            feat_fused = sum(currentFeatures, 3);
        case 'MajorityVote'
            for f=1:size(currentFeatures,3)
                % TODO: eventually norm the weight vector
                currentFeatures(:,:,f) = currentFeatures(:,:,f)*weights(f);
            end
            feat_sum = sum(currentFeatures, 3);
%             t = (size(currentFeatures,3)+1) / 2;
            t = sum(weights, 2)/2 + 1;     % Does not work so well?
            feat_fused = zeros(size(feat_sum));
            feat_fused(feat_sum>=t) = 1;
        case 'Staple'
            obs = create_obs('slice', [img_h img_w 1]);
            R=size(currentFeatures, 3);
            for r = 1:R
                obs = add_obs(obs, currentFeatures(:,:,r), 1, r);
            end
            
            if staplePrior_flag == -1   % -1 for svprior estimation
                [MV_est, svprior] = log_odds_majority_vote(obs, 1);
            else
                svprior = staplePrior_flag;
            end
            [Staple_est, Staple_W, Staple_theta] = STAPLE(obs, stapleEpsilon, svprior, stapleInit_flag, stapleCf);

            if saveProb
                feat_fused = Staple_W;
            else
                feat_fused = Staple_est;
            end
        case 'Stapler'
            obs = create_obs('slice', [img_h img_w 1]);
            R=size(currentFeatures, 3);
            for r = 1:R
                obs = add_obs(obs, currentFeatures(:,:,r), 1, r);
            end

            [~, svprior] = log_odds_majority_vote(obs, 1);
            % Use majority voting for truth
            % TODO: Check if better solution possible
            feat_sum = sum(currentFeatures, 3);
            t = (size(currentFeatures, 3) + 1) / 2;
            truth = zeros(size(feat_sum));
            truth(feat_sum>=t) = 1;
            [bias_theta tr_theta] = construct_theta_bias(obs, truth, obs, stapleCf);

            % run the STAPLER algorithm
            [Stapler_est, Stapler_W Stapler_theta] = STAPLER(obs, ...
                stapleEpsilon, svprior, stapleInit_flag, stapleCf, bias_theta);

            if saveProb
                feat_fused = Stapler_W(:,:,1,2);
            else
                feat_fused = Stapler_est;
            end
        case 'Collate'
            obs = create_obs('slice', [img_h img_w 1]);
            R=size(currentFeatures, 3);
            for r = 1:R
                obs = add_obs(obs, currentFeatures(:,:,r), 1, r);
            end
            
            if collatePrior_flag == -1   % -1 for svprior estimation
                [~, svprior] = log_odds_majority_vote(obs, 1);
            else
                svprior = collatePrior_flag;
            end

            [Collate_est, Collate_W, Collate_theta] = COLLATE(obs, ...
                collateEpsilon, svprior, collateInit_flag, collateAlphas, collateCvals);

            if saveProb
                % feat = cellfun(@(x) x(:,:,1,2), feature_COLLATE_prob2, 'UniformOutput', false);
                feat_fused = Collate_W(:,:,1,2);
            else
                feat_fused = Collate_est;
            end
        case 'SpatialStaple'
             % TODO
        case 'Simple'
            obs = create_obs('slice', [img_h img_w 1]);
            R=size(currentFeatures, 3);
            for r = 1:R
                obs = add_obs(obs, currentFeatures(:,:,r), 1, r);
            end
            
            Simple_est = SIMPLE(obs, simpleNumIterKeep, simpleAlpha, simplePerfType);
            feat_fused = Simple_est;    % no prob available for simple
        case 'LogOdds'
            obs = create_obs('slice', [img_h img_w 1]);
            R=size(currentFeatures, 3);
            for r = 1:R
                obs = add_obs(obs, currentFeatures(:,:,r), 1, r);
            end
            
            [LogOdds_Est, LogOdds_W] = log_odds_majority_vote(obs, logOddsDecayCoeff);

            if saveProb
                feat_fused = LogOdds_W;
            else
                feat_fused = logical(LogOdds_Est);
            end
        case 'LogOddsWeighted'
            obs = create_obs('slice', [img_h img_w 1]);
            R=size(currentFeatures, 3);
            for r = 1:R
                obs = add_obs(obs, currentFeatures(:,:,r), 1, r);
            end
            % Construct the weights
            loWeights = ones(img_h,img_w,1,size(currentFeatures,3));
            for f=1:size(currentFeatures,3)
                loWeights(:,:,1,f) = loWeights(:,:,1,f)*weights(f);
            end
            
            [LogOdds_Est, LogOdds_W] = log_odds_locally_weighted_vote(obs, loWeights, logOddsDecayCoeff);

            if saveProb
                feat_fused = LogOdds_W;
            else
                feat_fused = logical(LogOdds_Est);
            end
        otherwise
            fprintf(2, 'ERROR: Invalid Fusion Method: %s\n', fusionMethod);
            continue;
    end
    % Binarize images again using the median, TODO: Other threshold
    if doBinarize
        switch binarizationMethod
            case 'Mean'
                binThreshold = mean(feat_fused(feat_fused>0));
                feat_fused_bin = zeros(size(feat_fused));
                feat_fused_bin(feat_fused>binThreshold) = 1;
            case 'Median'
                binThreshold = median(feat_fused(feat_fused>0));
                feat_fused_bin = zeros(size(feat_fused));
                feat_fused_bin(feat_fused>binThreshold) = 1;
            case 'Threshold'
                % Use the threshold given
                feat_fused_bin = zeros(size(feat_fused));
                feat_fused_bin(feat_fused>binThreshold) = 1;
            case 'None'
                feat_fused_bin = feat_fused;
        end
        feat_fused = logical(feat_fused_bin);
    end
    % Morphologic operations to remove spurious pixel, only for binary
    if doBinarize && doBW
        feat_fused_bin = bwareaopen(feat_fused_bin, bWFirstPar);
        feat_fused_bin =~ bwareaopen(~feat_fused_bin, bWSecondPar);
    end
    if doBinarize && doMorphological
        feat_fused_bin = bwmorph(feat_fused_bin, bwOperation, bwTimes);
    end
    if doBinarize
        fusedFeatures{i} = feat_fused_bin;
    else
        fusedFeatures{i} = feat_fused;
    end
    if use_progress_bar
        if parallelProcessing
            parfor_progress;
        else
            updateStatus(i, nrImages, toc(starttime));
        end
    end
end

if use_progress_bar
    if parallelProcessing
        parfor_progress(0);
        delete(poolobj);    % Close parallel pool
    else
        ProgressBar.update('close');
    end
end

% Reshape back to original format
fusedFeatures = reshape(fusedFeatures, nr_users, nr_fingers, nr_images);
fprintf('Fusion of features finished.\n');
end
