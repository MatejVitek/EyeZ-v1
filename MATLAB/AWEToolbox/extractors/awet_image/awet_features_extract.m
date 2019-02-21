function [features] = awet_features_extract(I, I_annotations)
	% extract ear features for current image
	%
	% Input:
	%    I             = already preprocessed ear image
	%    I_annotations = annotation data for this image
	%
	% Output:
	%    features      = vector of ear features
 
    features = uint8(reshape(I, 1, []));
end