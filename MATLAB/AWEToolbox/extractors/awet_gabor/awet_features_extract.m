function [features] = awet_features_extract(I, I_annotations)
	% extract ear features for current image
	%
	% Input:
	%    I             = already preprocessed ear image
	%    I_annotations = annotation data for this image
	%
	% Output:
	%    features      = vector of ear features
 
    filter_bank = construct_Gabor_filters_PhD(8, 5, size(I));
	features = filter_image_with_Gabor_bank_PhD(I, filter_bank);
    mf = min(features);
    if (mf < 0)
        mf = -1 * mf;
    end
    features = features + mf;
end