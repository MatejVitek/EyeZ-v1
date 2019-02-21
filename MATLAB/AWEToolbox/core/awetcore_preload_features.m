function [ status, data ] = awetcore_preload_features(database, extractor)
    % check if current tmp of database exists in the temp folder
    % check if db-features combination exists in the calculated_features folder
    global awet;
	
    status = 0;
    data = [];
    calc_features_path = [awet.calculated_features_path, database.path, '-', extractor.path, awet.parameter_path, '.csv'];
    temp_db_path = [awet.temp_path, database.path, '.mat'];
    
    if (exist(calc_features_path, 'file'))
        
        addpath(awet.calculated_features_path);
        
        
        
        prompt = '\nPrecalculated features exist, attempt to load? Y/N [N]: ';
        yn = input(prompt, 's');
        if (~isempty(yn) && (isequal(yn, 'Y') || isequal(yn, 'y') || isequal(yn, '1')))
            data = readtable(calc_features_path);
            status = 1;
            awetcore_log('Precalculated features loaded! Feature extraction will be skipped!', 1);
        else
            status = 0;
            awetcore_log('Precalculated features NOT loaded. Feature extraction will execute.', 1);
        end
    elseif (exist(temp_db_path, 'file'))
        status = 2;
        addpath(awet.temp_path);
        data = load([database.path, '.mat']);
        % database will be read from this temp folder
        awetcore_log('Database temp exists, attempting to load ...', 1);
    end
end

