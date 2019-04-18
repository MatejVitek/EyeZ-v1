function [filename_root, filename_png, filename_fig, filename_pdf] = awetcore_check_dir(suffix, subfolder, folder_name_png, folder_name_fig, folder_name_pdf)
    global awet;
    dbname = ['P', num2str(awet.current_database.protocol), '_', awet.current_database.name];
    dirBase = [awet.results_path, awet.run_id, '-plots/'];    
    
    if ~exist(awet.results_path, 'dir')
        awetcore_log(['Folder ''', awet.results_path, ''' does not exist, creating it ...'], 2);
        mkdir(awet.results_path)
    end
    
    if ~exist(dirBase, 'dir')
		awetcore_log(['Folder ''', dirBase, ''' does not exist, creating it ...'], 2);
		mkdir(dirBase)
    end
    
    dir = [dirBase, subfolder, '/'];
    dir_png = [dir, folder_name_png, '/'];
    dir_fig = [dir, folder_name_fig, '/'];
    dir_pdf = [dir, folder_name_pdf, '/'];
    
    if ~exist(dir, 'dir')
		awetcore_log(['Folder ''', dir, ''' does not exist, creating it ...'], 2);
		mkdir(dir)
    end
    
    if ~exist(dir_png, 'dir')
		awetcore_log(['Folder ''', dir_png, ''' does not exist, creating it ...'], 2);
		mkdir(dir_png)
    end
    
    if ~exist(dir_fig, 'dir')
		awetcore_log(['Folder ''', dir_fig, ''' does not exist, creating it ...'], 2);
		mkdir(dir_fig)
    end
    
    if ~exist(dir_pdf, 'dir')
		awetcore_log(['Folder ''', dir_pdf, ''' does not exist, creating it ...'], 2);
		mkdir(dir_pdf)
    end
    
    filename_root = [dir, dbname,'-', suffix];
    filename_png = [dir_png, dbname,'-', suffix];
    filename_fig = [dir_fig, dbname,'-', suffix];
    filename_pdf = [dir_pdf, dbname,'-', suffix];
end
