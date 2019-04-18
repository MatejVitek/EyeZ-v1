function awet_init_extractor()
	% if you need to execute something before feature extractor starts use this function
	
	awetcore_log('RUNNING EXTRACTOR INIT', 1);
	if ~exist('vl_version', 'file')
		awetcore_log('vl_feat not yet installed, installing ...', 2);
		run('libraries/vlfeat-0.9.20/toolbox/vl_setup');
    end
    awetcore_inface_install();
end