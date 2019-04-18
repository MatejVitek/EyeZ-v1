function [I, I_annotations] = awet_preprocess(I, I_annotations)
	% preprocess applies preprocessing to 
	% each image in db
	% 
	% Input:
	%    I              = ear image
	%    I_annotations  = annotation data for the image
	% 
	% Output:
	%    I              = modified ear image
	%    I_annotations  = modified annotation data
    
    % all of this is already done in db_preprocessing
    %if (size(I, 3) == 3)
    %    I = rgb2gray(I);
    %end
    % this somehow makes LPQ perform better - need to reseach why though
    % also 20 by 20 is a sweet spot for CVLEDB#1
    %I = imresize(I, [40 40]);
    %[I, ~] = single_scale_retinex(I);
	
	I = uint8(I);
    
end