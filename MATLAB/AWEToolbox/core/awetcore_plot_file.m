function awetcore_plot_file(X, Y, f_suffix, x_label, y_label, plot_title, plot_handle)
    global awet;
    
    imgType = 'png';
    dir = [awet.results_path, awet.run_id, '-plots/'];
	filename = [dir, awet.current_database.path, '-', awet.current_extractor.path, '-', f_suffix, '.', imgType];
	
	if ~exist(awet.results_path, 'dir')
        awetcore_log(['Folder ''', awet.results_path, ''' does not exist, creating it ...'], 2);
        mkdir(awet.results_path)
    end
    
    if ~exist(dir, 'dir')
		awetcore_log(['Folder ''', dir, ''' does not exist, creating it ...'], 2);
		mkdir(dir)
    end
    
    figure('Visible','off')
    if isempty(plot_handle)    
        plot(X, Y);
        xlabel(x_label)
        ylabel(y_label)
        title(plot_title)
        saveas(gcf, filename);
    else
        saveas(plot_handle, filename);
    end
    %figure('Visible','on')
end