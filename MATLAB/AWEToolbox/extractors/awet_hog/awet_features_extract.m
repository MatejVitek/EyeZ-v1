function [features] = awet_features_extract(I, I_annotations)
	% extract ear features for current image
	%
	% Input:
	%    I             = already preprocessed ear image
	%    I_annotations = annotation data for this image
	%
	% Output:
	%    features      = vector of ear features
 
    %  'CellSize'     A 2-element vector that specifies the size of a HOG cell
%                 in pixels. Select larger cell sizes to capture large
%                 scale spatial information at the cost of loosing small
%                 scale detail.
%                 
%                 Default: [8 8]
%
%  'BlockSize'    A 2-element vector that specifies the number of cells in
%                 a block. Large block size values reduce the ability to
%                 minimize local illumination changes.
%
%                 Default: [2 2]
%
%  'BlockOverlap' A 2-element vector that specifies the number of
%                 overlapping cells between adjacent blocks. Select an
%                 overlap of at least half the block size to ensure
%                 adequate contrast normalization. Larger overlap values
%                 can capture more information at the cost of increased
%                 feature vector size. This property has no effect when
%                 extracting HOG features around point locations.
% 
%                 Default: ceil(BlockSize/2)
%                  
%  'NumBins'      A positive scalar that specifies the number of bins in
%                 the orientation histograms. Increase this value to encode
%                 finer orientation details.
%                 
%                 Default: 9
%
%  'UseSignedOrientation' A logical scalar. When true, orientation
%                         values are binned into evenly spaced bins
%                         between -180 and 180 degrees. Otherwise, the
%                         orientation values are binned between 0 and
%                         180 where values of theta less than 0 are
%                         placed into theta + 180 bins. Using signed
%                         orientations can help differentiate light to
%                         dark vs. dark to light transitions within
%                         an image region.
%
%                         Default: false
    properties = struct();
    %[features, ~] = extractHOGFeatures(I, 'CellSize', [8 8], 'BlockSize', [2 2], 'BlockOverlap', ceil([2 2]/2), 'NumBins', 9, 'UseSignedOrientation', false);
    %[features, ~] = extractHOGFeatures(I, 'CellSize', [8 8], 'BlockSize', [8 8], 'BlockOverlap', ceil([8 8]/2), 'NumBins',18, 'UseSignedOrientation', false);
    [features, ~] = extractHOGFeatures(I, 'CellSize', [8 8], 'BlockSize', [2 2], 'BlockOverlap', [1 1], 'NumBins',18, 'UseSignedOrientation', false);
end