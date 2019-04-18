function [features] = awet_features_extract(I, I_annotations)
	% extract ear features for current image
	%
	% Input:
	%    I             = already preprocessed ear image
	%    I_annotations = annotation data for this image
	%
	% Output:
	%    features      = vector of ear features
 
    min = 1;
    max = 10000;
    len = 100;
    features = (round((max-min).*rand(len,1) + min))';
end