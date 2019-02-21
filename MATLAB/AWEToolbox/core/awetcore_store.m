function awetcore_store(type, data, database, extractor)
    global awet;
	
    if (isequal(type, 1) && awet.store_features)
        % store features
        if ~exist(awet.calculated_features_path, 'dir')
            awetcore_log(['Folder ''', awet.calculated_features_path, ''' does not exist, creating it ...'], 2);
            mkdir(awet.calculated_features_path)
        end
        calc_features_path = [awet.calculated_features_path, database.path, '-', extractor.path, awet.parameter_path, '.csv'];
        awetcore_log('Storing data ...', 2);
        writetable(data, calc_features_path);
        awetcore_log('Finished storing data.', 2);
        
        % TODO: Store arff format
        
    elseif (isequal(type, 2))
        % store db
        if ~exist(awet.temp_path, 'dir')
            awetcore_log(['Folder ''', awet.temp_path, ''' does not exist, creating it ...'], 2);
            mkdir(awet.temp_path)
        end
        temp_db_path = [awet.temp_path, database.path, '.mat'];
        db = data.db; %#ok<NASGU>
        annotations = data.annotations; %#ok<NASGU>
        awetcore_log('Storing data ...', 2);
        save(temp_db_path, 'db', 'annotations');
        awetcore_log('Finished storing data.', 2);
    end
end

