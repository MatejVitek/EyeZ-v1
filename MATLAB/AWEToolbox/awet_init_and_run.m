function awet_init_and_run(databases, preprocessor, extractors, models, crossvalind, log_level, protocol_type, norm_features, compressed_evaluation, use_phd, norm_mode)
    addpath('core');
    addpath('libraries/jsonlab');
    addpath('libraries/histogram_distance');

    global awet;
    awet.VERSION = '1.4';
    awet.databases_path = 'databases/';
    awet.preprocessor_path = 'preprocessors/';
    awet.extractors_path = 'extractors/';
    awet.models_path = 'models/';
    awet.file_endings = {'png', 'jpg', 'bmp'};
    awet.annotations_file_name = 'annotations.json';
    awet.temp_path = '_output/temp/';
    awet.results_path = '_output/results/';
    awet.calculated_features_path = '_output/calculated_features/';
    awet.run_id = datestr(now,'yyyymmdd-HHMM-SSFFF');
    awet.results_header_set = 0;
    awet.distribution_files = [
        struct('train', 'train.txt', 'test', 'test.txt', 'classes', 'classes.txt', 'distr', 'distr.mat')
    ];
    awet.plots = struct();
% Nice colors
%     awet.plot_colors = [ ...
%         0.2, 0.133, 0.533; ...
%         0.533, 0.8, 0.933; ...
%         0.267, 0.667, 0.6; ...
%         0.067, 0.467, 0.2; ...
%         0.6, 0.6, 0.2; ...
%         0.867, 0.8, 0.467; ...
%         0.8, 0.4, 0.467; ...
%         0.533, 0.133, 0.333; ...
%         0.667, 0.267, 0.6; ...
%         0.5, 0.5 0.5 ];
    
    awet.plot_colors = [ ...
%         0.2, 0.133, 0.533; ...
        0.533, 0.8, 0.933; ...
        0.267, 0.667, 0.6; ...
        0.067, 0.467, 0.2; ...
        0.6, 0.6, 0.2; ...
        0.867, 0.8, 0.467; ...
        0.8, 0.4, 0.467; ...
        0.533, 0.133, 0.333; ...
        0.667, 0.267, 0.6; ...
        0.5, 0.5 0.5 ];

% Automatically different colors
%     addpath(genpath('libraries/colors'));
%     awet.plot_colors = distinguishable_colors(20);

% Distinguishable colors
% http://www.edwardtufte.com/bboard/q-and-a-fetch-msg?msg_id=0000HT
%     awet.plot_colors = [ ...
%         0 0 0; ...
%         0.90 0.60 0; ...
%         0.35 0.70 0.90; ...
%         0.0 0.60 0.50; ...
%         0.95 0.90 0.25; ...
%         0 0.45 0.70; ...
%         0.80 0.40 0; ...
%         0.80 0.60 0.70; ...
%         0.50 0.50 0.50 ];

    awet.databases_default = [
        %struct('path', 'cvledb', 'name', 'CVLEDB', 'protocol', 1)
		struct('path', 'awe', 'name', 'AWE', 'protocol', 2)
		%struct('path', 'awe', 'name', 'AWE', 'protocol', 3)
    ];
    awet.preprocessor_default = struct('path', 'awet_basic', 'name', 'Basic preprocessing');
	awet.extractors_default = [
		struct('path', 'awet_lbp',     'name', 'LBP',  'distance', 'chi', 'bulk_postprocess', false, 'parameters', struct())
		struct('path', 'awet_bsif',    'name', 'BSIF', 'distance', 'chi', 'bulk_postprocess', false, 'parameters', struct())
		struct('path', 'awet_lpq',     'name', 'LPQ', 'distance', 'chi', 'bulk_postprocess', false, 'parameters', struct())
		struct('path', 'awet_rilpq',   'name', 'RILPQ','distance', 'chi', 'bulk_postprocess', false, 'parameters', struct())    
		struct('path', 'awet_poem',    'name', 'POEM', 'distance', 'chi', 'bulk_postprocess', false, 'parameters', struct())
		struct('path', 'awet_hog',     'name', 'HOG',  'distance', 'chi', 'bulk_postprocess', false, 'parameters', struct())
		struct('path', 'awet_dsift',   'name', 'DSIFT','distance', 'chi', 'bulk_postprocess', false, 'parameters', struct())
		struct('path', 'awet_gabor',   'name', 'Gabor','distance', 'cosine', 'bulk_postprocess', false, 'parameters', struct())
	];
	awet.models_default = [
		struct('path', '', 'name', '', 'mode', 'none')
		%struct('path', 'awet_knn', 'name', 'K-nearest neighbor', 'mode', 'identification')
		%struct('path', 'awet_svm', 'name', 'Support vector machine', 'mode', 'identification')
	];
    awet.crossvalind_default = struct('method', 'Kfold', 'factor', 5, 'train_or_test', 'train');
    awet.log_level_default = 1;
    awet.protocol_type_default = 1;
    awet.norm_features_default = 1;
    awet.compressed_evaluation_default = 1;
    awet.use_phd_default = 1;
    awet.norm_mode_default = 2;
    
    if isempty(databases)
        awet.databases = awet.databases_default;
    else
        awet.databases = databases;
    end
    
    if isempty(preprocessor)
        awet.preprocessor = awet.preprocessor_default;
    else
        awet.preprocessor = preprocessor;
    end
    
    if isempty(extractors)
        awet.extractors = awet.extractors_default;
    else
        awet.extractors = extractors;
    end
    
    if isempty(models)
        awet.models = awet.models_default;
    else
        awet.models = models;
    end
    
    if isempty(log_level)
        awet.log_level = awet.log_level_default;
    else
        awet.log_level = log_level;
    end
    
    if isempty(crossvalind)
        awet.crossvalind = awet.crossvalind_default;
    else
        awet.crossvalind = crossvalind;
    end
    
    if isempty(protocol_type)
        awet.protocol_type = awet.protocol_type_default;
    else
        awet.protocol_type = protocol_type;
    end
    
    if isempty(norm_features)
        awet.norm_features = awet.norm_features_default;
    else
        awet.norm_features = norm_features;
    end
    
    if isempty(compressed_evaluation)
        awet.compressed_evaluation = awet.compressed_evaluation_default;
    else
        awet.compressed_evaluation = compressed_evaluation;
    end
    
    if isempty(use_phd)
        awet.use_phd = awet.use_phd_default;
    else
        awet.use_phd = use_phd;
    end
    
    if isempty(norm_mode)
        awet.norm_mode = awet.norm_mode_default;
    else
        awet.norm_mode = norm_mode;
    end
    
    combinations = numel(awet.databases) * numel(awet.extractors) * numel(awet.models);
    awetcore_log('logo', 0);
    awetcore_log(['Start time is ', datestr(now), '.'], 0);
    awetcore_log(['Run ID is ', awet.run_id, '.'], 0);
    i = 1;
    
    OVERRIDE_PROMPT = 0;
    if OVERRIDE_PROMPT == 0
        prompt = '\nDo you want to store calculated ear features? Y/N [N]: ';
        yn = input(prompt, 's');
        if (~isempty(yn) && (isequal(yn, 'Y') || isequal(yn, 'y') || isequal(yn, '1')))
            awetcore_log('Ear features will be stored to CSV file.', 0);
            awet.store_features = true;
        else
            awetcore_log('Ear features will NOT be stored.', 0);
            awet.store_features = false;
        end
    else
        awetcore_log('Ear features stored.', 0);
        awet.store_features = true;
    end
    
    startTime = tic;
    
    awetcore_log(['\nUsing "', awet.preprocessor.name, '" on all DB images on all DBs.'], 0);
    
    warning('off', 'MATLAB:rmpath:DirNotFound');
    rmpath(genpath(awet.preprocessor_path));
    addpath([awet.preprocessor_path, awet.preprocessor.path]);
    warning('on', 'MATLAB:rmpath:DirNotFound');
    
    for id = 1:numel(awet.databases)
        awet.current_database = awet.databases(id);
        database = awet.current_database;
        
        %% READ DB
        awetcore_log(['\n_____________________________________________ Loading ', database.name,'\n'], 0);
        awetcore_log('BEGIN\tPreload ', 1);
        [status, data] = awetcore_preload_db(database);
        awetcore_log('END\t\tPreload ', 1); 
    
        if (isequal(status, 1))
            awetcore_log('USING PRELOADED DB', 1);
            db = data.db;
            annotations = data.annotations;
        else
            awetcore_log(['BEGIN\tReading DB ', database.path], 1); 
            [db, annotations] = awetcore_database_load(database.path);
            [db, annotations] = awetcore_db_preprocess_all(db, annotations);
            awetcore_store(2, struct('db', db, 'annotations', annotations), database);
            awetcore_log('END\t\tReading DB', 1);
        end
        
        %% PREPARE DIVISIONS
        % The following function divides db into folds and returns indices
        divisions = awetcore_divider(db.class);
        
        %% LOOP THROUGH EXTRACTORS and MODELS
        % LOOP OVER EXTRACTORS
        for ie = 1:numel(awet.extractors)
            awet.current_extractor = awet.extractors(ie);
            % LOOP OVER ALL COMBINATIONS OF PARAMETERS
            
            allParamCombination = {''};
            if (isfield(awet.current_extractor, 'parameters'))
                allParam = awet.current_extractor.parameters;            
                allParamNames = fieldnames(allParam);
                arrParamInx = ones(size(allParamNames));
                allParamCombination = getParamCombinations({}, allParam, allParamNames, arrParamInx, 1);                
            end
            sentinel = 0;
            for paramCombination = allParamCombination
                awet.current_parameter_set = paramCombination{:};
                awet.parameter_path = awetcore_generate_param_name(paramCombination{:});
            
                % to ignore param count
                if sentinel > 0
                    i = i - 1;
                end

                % LOOP OVER ALL MODELS
                for im = 1:numel(awet.models)
                    awet.current_model = awet.models(im);

                    warning('off', 'MATLAB:rmpath:DirNotFound');
                    rmpath(genpath(awet.extractors_path));
                    rmpath(genpath(awet.models_path));
                    rmpath(genpath(awet.databases_path));
                    warning('on', 'MATLAB:rmpath:DirNotFound');

                    addpath([awet.extractors_path, awet.current_extractor.path]);
                    addpath([awet.models_path, awet.current_model.path]);
                    addpath([awet.databases_path, awet.current_database.path]);

                    awet.ident_or_verif = isequal(awet.current_model.mode, 'verification');
                    awet.bulk_postprocess = awet.current_extractor.bulk_postprocess;

                    awet_run(awet.current_database, awet.current_extractor, awet.current_parameter_set, awet.current_model, i, combinations, db, annotations, divisions);

                    i = i + 1;
                end % models
                sentinel = 1;
            end % params
        end % extractors
    end % datasets
    awetcore_log([
         '\n_____________________________________________ END\n' ...
        ,'\n\nRun ID was ', awet.run_id, '.' ...
        ,'\nEnd time is ', datestr(now), '.' ...
    ], 0);
    %toc(startTime);
    tocEnd = toc(startTime);
    tocEndPrint = fprintf('It took %d minutes and %f seconds.\n', floor(tocEnd/60), rem(tocEnd,60));
    awetcore_log([tocEndPrint, '\n'], 0);
end

function paramData = getParamCombinations(paramData, allParam, allParamNames, arrParamInx, pos)
    % get current
    tmpp = setParamData(allParam, allParamNames, arrParamInx);
    paramData{size(paramData, 2)+1} = tmpp;

    % go recursevely through all params
    lim = size(arrParamInx, 1);
    for i = pos:lim
        % dive with one more if possible
        name = allParamNames(i);
        currParamData = allParam.(name{:});
        if size(currParamData, 2) > arrParamInx(i)
            newArrParamInx = arrParamInx;
            newArrParamInx(i, 1) = newArrParamInx(i, 1) + 1;
            newPos = i;
            tmpp = getParamCombinations(paramData, allParam, allParamNames, newArrParamInx, newPos);
            paramData = tmpp;
        end
    end
    
end

function paramData = setParamData(allParam, allParamNames, arrParamInx)
    paramData = struct();
    for i = 1:size(allParamNames, 1)
        paramName = allParamNames(i);
        paramData.(paramName{:}) = allParam.(paramName{:})(arrParamInx(i));
    end
end
