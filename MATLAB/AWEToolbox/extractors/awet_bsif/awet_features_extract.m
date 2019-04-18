function [features] = awet_features_extract(I, I_annotations)
	% extract ear features for current image
	%
	% Input:
	%    I             = already preprocessed ear image
	%    I_annotations = annotation data for this image
	%
	% Output:
	%    features      = vector of ear features    
    global awet;
    filename = [awet.extractors_path, awet.current_extractor.path, '/texturefilters/ICAtextureFilters_11x11_8bit'];
	load(filename, 'ICAtextureFilters');
    
	features = compute_bsif_features(I, ICAtextureFilters);
end