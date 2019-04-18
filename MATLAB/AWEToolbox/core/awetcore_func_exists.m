function status = awetcore_func_exists(func, fatal)
	status = exist(func, 'file');
	if ~status
		if ~fatal
			awetcore_log(['SKIPPING: function ', func, ' not present'], 2);
		else
			awetcore_log(['FATAL ERROR: function ', func, ' not present'], 0);
		end
	end
end

