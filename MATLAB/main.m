%% Config
% Source directory for images
src_dir = '/hdd/EyeZ/SBVPI/Subsets/Rot ScleraNet/stage2_ungrouped/';

% Matching algorithm
%alg_dir = 'Miura';
alg_dir = 'B-COSFIRE';
old_dir = cd(alg_dir);

% Directory to save results to
save_dir = strcat('/hdd/EyeZ/Segmentation/Results/Vessels/', alg_dir);
overwrite = false;

% Evaluate Miura algorithms
%for alg_name = {'MC', 'RLTGS'}
%	evaluate_alg_pair(alg_name{1}, src_dir, save_dir);
%end

% Evaluate a single algorithm
evaluate_alg(@my_pred, src_dir, strcat(save_dir, '/'), overwrite);

cd(old_dir);

%% Evaluate binarised and normalised algorithm version
function evaluate_alg_pair(alg_name, src_dir, save_dir)
	save = strcat(save_dir, '_', alg_name, '/');
	alg = @(image, sclera)my_pred(image, sclera, alg_name);
	evaluate_alg(alg, src_dir, save);
	
	save = strcat(save_dir, '_', alg_name, '_norm/');
	alg = @(image, sclera)my_pred(image, sclera, alg_name, 'norm');
	evaluate_alg(alg, src_dir, save);
end

%% Evaluate a single algorithm
function evaluate_alg(alg, src_dir, save, overwrite)
	if ~exist(save, 'dir')
		mkdir(save);
	end
	
	set = imageSet(src_dir);
	k = 0;
	% Reduce number of workers below if too much memory is being used
	parfor (id_image = 1:set.Count, 24)
		% If it's a mask image, skip it
		[path, basename, ~] = fileparts(set.ImageLocation{id_image});
		if regexp(basename, '\d+[LR]_[lrsu]_\d+_')
			continue
		end

		% Evaluate a single image
		score = evaluate(path, basename, alg, save, overwrite);

		% If sclera mask was not found, skip
		if score == -1
			continue
		end

		% Otherwise save the result
		k = k + 1;
		%results(k) = score;
	end
	disp(['Found ' num2str(k) ' images.']);
	%results = results(1:k);
end

%% Load an image and its corresponding sclera prediction and vessels GT mask and evaluate the algorithm on it
function score = evaluate(path, basename, alg, save, overwrite)
	sclera_file = strcat(path, '/', basename, '_sclera.png');
	vessels_file = strcat(path, '/', basename, '_vessels.png');
	
	disp(basename);
	if ~isfile(sclera_file)
		disp('404: Sclera not found');
		score = -1;
		return
	end
	vessels_exist = isfile(vessels_file);
	
	% Load base image
	image = im2double(imread(strcat(path, '/', basename, '.jpg')));
	s = size(image);
	
	% Load masks
	sclera = im2double(rgb2gray(imread(sclera_file)));
	if vessels_exist
		vessels = im2double(rgb2gray(imread(vessels_file)));
	end
	
	if overwrite || ~isfile(strcat(save, basename, '.png'))
		% Run the matching algorithm
		prediction = alg(image, sclera);
		% Save prediction
		imwrite(prediction, strcat(save, basename, '.png'));
	end
	if vessels_exist
		% Save GT mask
		imwrite(vessels, strcat(save, basename, '_gt.png'));
	end
	
	% TODO: Calculate and return error rate
	score = 0;
end

%% Segment the part of the RGB image where grayscale mask has a value > threshold
function image = segment(image, mask, threshold)
	if nargin < 3
		threshold = 0.5;
	end
	mask = mask > threshold;
	mask = cat(3, mask, mask, mask);
	image = image .* mask;
end
