function [filename] = awetcore_check_dir_raw(subfolder)
    global awet;
    
    dirBase = [awet.results_path, awet.run_id, '-plots/'];    
    protname = ['P', num2str(awet.current_database.protocol)];
    dbname = awet.current_database.name;
    
    if ~exist(awet.results_path, 'dir')
        awetcore_log(['Folder ''', awet.results_path, ''' does not exist, creating it ...'], 2);
        mkdir(awet.results_path)
    end
    
    if ~exist(dirBase, 'dir')
		awetcore_log(['Folder ''', dirBase, ''' does not exist, creating it ...'], 2);
		mkdir(dirBase)
    end
    
    dir = [dirBase, subfolder, '/'];    
    if ~exist(dir, 'dir')
		awetcore_log(['Folder ''', dir, ''' does not exist, creating it ...'], 2);
		mkdir(dir)
    end
    
    dir = [dir, protname, '/'];    
    if ~exist(dir, 'dir')
		awetcore_log(['Folder ''', dir, ''' does not exist, creating it ...'], 2);
		mkdir(dir)
    end
    
    dir = [dir, dbname, '/'];    
    if ~exist(dir, 'dir')
		awetcore_log(['Folder ''', dir, ''' does not exist, creating it ...'], 2);
		mkdir(dir)
    end
    
    filename = [dir, awet.current_extractor.name, awet.parameter_path];
end
