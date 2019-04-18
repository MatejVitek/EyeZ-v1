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
% This function provides a percentage thresholding based binarization for
% input images, also supporting a pixel mask
%
% Parameters:
%  img  		- The input image which should be binarized
%  mask 		- A mask which pixel to use inside the input imge
%  proportion 	- Which proportion of pixels should be white/black
%
% Returns:
%  binarized_image  - The binarized image according to the set proportion
%  threshold        - The threshold corresponding to the proportion
% *************************************************************************
function [binarized_image, threshold] = percentage_thesholding(img, mask, proportion)
	mask_1 = mask(:);
	img_1 = img(:);
	img_2=img_1(mask_1);
	if isempty(img_2)
		binarized_image = mask;
		threshold = NaN;
	else
		data_sorted = sort(img_2);
		
		thresh_ind = round(proportion * numel(data_sorted));
		threshold = data_sorted(thresh_ind);
		
		binarized_image=zeros(size(img));
		binarized_image(img>threshold) = 1;
	end
end
