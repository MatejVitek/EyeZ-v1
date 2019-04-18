% *************************************************************************
% This file is part of the feature level fusion framework for finger vein
% recognition (MATLAB implementation). 
%
% Reference:
% Advanced Variants of Feature Level Fusion for Finger Vein Recognition 
% C. Kauba, E. Piciucco, E. Maiorana, P. Campisi and A. Uhl 
% In Proceedings of the International Conference of the Biometrics Special 
% Interest Group (BIOSIG'16), pp. 1-12, Darmstadt, Germany, Sept. 21 - 23
%
% Authors: Christof Kauba <ckauba@cosy.sbg.ac.at> and 
%          Emanuela Piciucco <emanuela.piciucco@stud.uniroma3.it>
% Date:    31th August 2016
% License: Simplified BSD License
%
% 
% Description:
% This function provides a kmeans based binarization method for images
%
% Parameters:
%  img		  - The input image which should be binarized
%  mask 	  - An optional mask which pixels should be regarded
%
% Returns:
%  bin_imag   - The output binarized image using kmeans
% *************************************************************************
function bin_image = kmeans_binarize(img, mask)
	if nargin == 1		% no mask given
		md  = median(img(img>0));
		c1  = mean(img(img < md));
		c2  = mean(img(img >= md));  		
		idx = kmeans(img(:), 2, 'start', [c1;c2]);
		bin_image = reshape((idx==2), size(img));
	elseif nargin == 2	% mask given
		fvr = logical(mask);
		img_region = img(fvr);
		%md  = median(img_region);
		md  = mean(img_region);
		c1  = mean(img_region(img_region < md));
		c2  = mean(img_region(img_region >= md));    
		idx = kmeans(img(fvr), 2, 'start', [c1;c2]);
		bin_image = zeros(size(img));
		bin_image(fvr) = (idx==2);    
	else    
		fprintf('Either no image or to many parameters given.\n')
	end
end
