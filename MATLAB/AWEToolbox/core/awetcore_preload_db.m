function [ status, data ] = awetcore_preload_db(database)
    % check if current tmp of database exists in the temp folder
    % check if db-features combination exists in the calculated_features folder
    global awet;
	
    status = 0;
    data = [];
    temp_db_path = [awet.temp_path, database.path, '.mat'];
    
    if (exist(temp_db_path, 'file'))
        status = 1;
        addpath(awet.temp_path);
        data = load([database.path, '.mat']);
        % database will be read from this temp folder
        awetcore_log('Database temp exists, attempting to load ...', 1);
    end
end

