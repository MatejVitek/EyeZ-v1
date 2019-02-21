function [features] = awet_features_extract(I, I_annotations)
	% extract ear features for current image
	%
	% Input:
	%    I             = already preprocessed ear image
	%    I_annotations = annotation data for this image
	%
	% Output:
	%    features      = vector of ear features    
    
    mapping = getmapping(8,'u2');
	features = compute_lbp_features(I, mapping,[16 16]);
end