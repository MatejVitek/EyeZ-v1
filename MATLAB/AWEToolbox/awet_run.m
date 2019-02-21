function awet_run(database, extractor, parameterSet, model, i, combinations, db, annotations, divisions)
    global awet;
	
    tic;
    
    awetcore_log(['\n_____________________________________________ ', ...
        num2str(i),'/', num2str(combinations),'\n'], 0);
    awetcore_log(['Protocol\t', num2str(database.protocol)], 0);
    awetcore_log(['Database\t', database.name], 0);
    awetcore_log(['Extractor\t', extractor.name], 0);
    awetcore_log(['Parameters\t', awetcore_generate_param_name(parameterSet)], 0);
    awetcore_log(['Model\t\t', model.name], 0);
    awetcore_log(['Crossval\t', awet.crossvalind.method, ' ' ...
        , num2str(awet.crossvalind.factor), '\n'], 0);

    % PRELOAD
    awetcore_log('BEGIN\tPreload ', 1);
    [status, data] = awetcore_preload_features(database, extractor);
    awetcore_log('END\t\tPreload ', 1); 
    
    if (isequal(status, 1))
        awetcore_log('USING PRELOADED FEATURES', 1);
        features = data;
    else
        % OPTIONAL EXTRACTOR INIT
        if awetcore_func_exists('awet_init_extractor', 0)
            awet_init_extractor();
        end

        % PREPROCESSING
        awetcore_log('BEGIN\tpreprocess', 1);
        [db, annotations] = awetcore_preprocess_all(db, annotations);
        awetcore_log('END\t\tpreprocess', 1);

        % FEATURE EXTRACTION
        % if ~awet.bulk_postprocess postprocess is called automatically
        awetcore_log('BEGIN\tfeatures extract', 1);
        features = awetcore_features_extract_all(db, annotations);
        awetcore_log('END\t\tfeatures extract', 1);

        % POSTPROCESSING
        if (awet.bulk_postprocess)
            awetcore_log('BEGIN\tpostprocess BULK', 1);
            features = awetcore_postprocess_all(features);
            awetcore_log('END\tpostprocess BULK', 1);
        end
        
        awetcore_store(1, features, database, extractor);
    end
    
    % OPTIONAL MODEL INIT
    if awetcore_func_exists('awet_init_model', 0)
        awet_init_model();
    end

    % EVALUATION
    awetcore_log('BEGIN\tevaluation', 1);
%     if isequal(model.name, '')
%         awetcore_log('DISTANCE MODE', 1);
%         awetcore_evaluate_by_distance(features);
%     else
%         awetcore_log('CLASSIFIER MODE', 1);
%         %[targets, outputs] = awetcore_evaluate(features);
%         [targets, outputs, scores] = awetcore_evaluate_constant(features);
%     end
    % TODONOW protocol handler zgolj prebere razdelitev iz globalne
    % spremenljivke
    [features_divided, classes] = awetcore_protocol_handler(features, divisions);
    [distances, mMin, mMax] = awetcore_evaluate_constant(features_divided, classes, extractor.distance);
	awetcore_log('END\tevaluation\n', 1);

    % OUTPUT AND VIZUALIZE THE RESULTS
    %awetcore_vizualize(targets, outputs, scores);
    awetcore_vizualize_roc_cmc(distances, mMin, mMax);
    
    toc;
end