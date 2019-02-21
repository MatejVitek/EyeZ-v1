function [db, annotations] = awet_database_preprocess(db, annotations)
	% preprocess applies preprocessing to 
	% each image in db for each db
	% 
	% Input:
	%    I              = ear image
	%    I_annotations  = annotation data for the image
	% 
	% Output:
	%    I              = modified ear image
	%    I_annotations  = modified annotation data

	% TODO calc average
    
    if (exist('single_scale_retinex','file'))
        awetcore_log('INFace is installed.\n', 2);
    else
        awetcore_log('INFace NOT installed. Attempting install ...\n', 2);
        
        addpath(genpath('libraries/INface_tool'));
        
        awetcore_log('INFace installed.\n', 2);
    end
    
    RES_factor = 100;
    
    strs = zeros(size(db, 1), RES_factor*RES_factor);
    for i = 1:size(db, 1)
		I = db(i, :).image{1};
		if (size(annotations) > 0)
			annotations_row = annotations(i, :);
		else
			annotations_row = [];
		end
		
		if (size(I, 3) == 3)
				I = rgb2gray(I);
		end
		I = imresize(I, [RES_factor RES_factor]);
        I = histeq(I);
		I = double(I);
		
        %I = single_scale_retinex(I);
        
        strs(i, :) = I(:);
        db(i, :).image{1} = I;
        
		%imwrite(uint8(I), ['_output/normalized_images/', num2str(i), '.png']);
		% annotations currently unused
    end
    [~, ns] = unique(strs, 'rows');
    to_del = setdiff(1:size(strs, 1), ns);
    db(to_del, :) = [];
    awetcore_log(['Number of duplicates in DB ignored: ', num2str(length(to_del))], 1);
    
end