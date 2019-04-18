function awetcore_log_file(disp_arr)
    global awet;
	filename = [awet.results_path, awet.run_id, '.txt']
	
	if ~exist(awet.results_path, 'dir')
		awetcore_log(['Folder ''', awet.results_path, ''' does not exist, creating it ...'], 2);
		mkdir(awet.results_path)
	end
	
    fid = fopen(filename, 'at');
	disp_arr = [disp_arr, '\n'];
	fprintf(fid, disp_arr);
	fclose(fid);
end