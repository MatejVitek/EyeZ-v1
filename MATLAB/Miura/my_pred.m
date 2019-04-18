function prediction = my_pred(image, sclera, algorithm, threshold)
	if nargin < 3
		algorithm = 'MC';
	end
	
	switch algorithm
		case 'MC'
			prediction = mc(image, sclera);
		case 'RLT'
			prediction = rlt(image, sclera);
			prediction = prediction(:, :, 1);
		case 'RLTGS'
			prediction = rlt(rgb2gray(image), sclera);
		otherwise
			disp('Unknown algorithm');
	end
	
	if nargin < 4
		threshold = median(prediction(prediction > 0));
	end

	if strcmp(threshold, 'norm')
		% Normalise the vein image
		mn = min(prediction(:));
		mx = max(prediction(:));
		prediction = (prediction - mn) / (mx - mn);
	else
		% Binarise the vein image
		prediction = prediction >= threshold;
	end
end

%% Extract veins using maximum curvature method
function prediction = mc(image, sclera)
	sigma = 3;
	prediction = miura_max_curvature(rgb2gray(image), sclera, sigma);
end

%% Extract veins using repeated line tracking method
function prediction = rlt(image, sclera)
	max_iterations = 3000;
	r=1;
	W=17;
	prediction = miura_repeated_line_tracking(image, sclera, max_iterations, r, W);
end
