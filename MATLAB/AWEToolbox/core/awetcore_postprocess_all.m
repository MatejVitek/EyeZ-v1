function features = awetcore_postprocess_all(features)
	% do postprocessing on all the feature vectors
	% 
	% Input:
	%    features = array of ear features for each image in db or for all
	%    images. This depends on bulk_postprocess attribute
	% 
	% Output:
	%    features = modified features
     
	if awetcore_func_exists('awet_postprocess', 0)
		features = awet_postprocess(features);
	end
end