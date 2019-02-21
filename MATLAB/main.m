%% Config
% Source directory for images
src_dir = '/hdd/EyeZ/Rot/SBVPI/SBVP_vessels/';

% Source directory for SegNet sclera mask predictions
sclera_dir = '/hdd/EyeZ/Rot/Segmentation/Results/segnet/';

% Matching algorithm
alg_dir = 'Miura';
addpath(strcat('./', alg_dir));

% Directory to save results to
save_dir = strcat('/hdd/EyeZ/Rot/Segmentation/Results/', alg_dir);

% Evaluate algorithms
for alg_name = {'MC', 'RLT', 'RLTGS'}
	evaluate_alg_pair(alg_name{1}, src_dir, sclera_dir, save_dir);
end

%% Evaluate binarised and normalised algorithm version
function evaluate_alg_pair(alg_name, src_dir, sclera_dir, save_dir)
	save = strcat(save_dir, '_', alg_name, '/');
	alg = @(image, sclera)(my_pred(image, sclera, alg_name));
	evaluate_alg(alg, src_dir, sclera_dir, save);
	
	save = strcat(save_dir, '_', alg_name, '_norm/');
	alg = @(image, sclera)(my_pred(image, sclera, alg_name, 'norm'));
	evaluate_alg(alg, src_dir, sclera_dir, save);
end

%% Evaluate a single algorithm
function evaluate_alg(alg, src_dir, sclera_dir, save)	
	if ~exist(save, 'dir')
		mkdir(save);
	end

	set = imageSet(src_dir, 'recursive');
	%results = zeros(1, sum(vertcat(set.Count)));
	k = 0;
	
	parfor (id_images = 1:length(set), 24)
		% Iterate over all images of current ID
		for image = set(id_images).ImageLocation
			% If it's a mask image, skip it
			[path, basename, ~] = fileparts(image{1});
			if regexp(basename, '\d+[LR]_[lrsu]_\d+_')
				continue
			end

			% Evaluate a single image
			score = evaluate(path, basename, alg, sclera_dir, save);

			% If a mask was not found, skip
			if score == -1
				continue
			end

			% Otherwise save the result
			k = k + 1;
			%results(k) = score;
		end
	end
	disp(['Found ' num2str(k) ' images.']);
	%results = results(1:k);
end

%% Load an image and its corresponding sclera prediction and vessels GT mask and evaluate the algorithm on it
function score = evaluate(path, basename, alg, sclera_dir, save)
	sclera_file = strcat(sclera_dir, basename, '.png');
	vessels_file = strcat(path, '/', basename, '_vessels.png');
	
	disp(basename);
	if ~isfile(sclera_file)
		disp('404: Sclera not found');
		score = -1;
		return
	end
	vessels_exist = isfile(vessels_file);
	
	% Load base image
	image = im2double(imread(strcat(path, '/', basename, '.JPG')));
	s = size(image);
	
	% Load masks
	sclera = imresize(im2double(rgb2gray(imread(sclera_file))), s(1:2));
	if vessels_exist
		vessels = im2double(imread(vessels_file));
	
		% Transform vessels to a proper mask
		eps = 0.05;
		r = vessels(:, :, 1);
		g = vessels(:, :, 2);
		b = vessels(:, :, 3);
		vessels = r < eps & g > 1 - eps & b < eps;
	end
	
	% Run the matching algorithm
	prediction = alg(image, sclera);
	
	% Save prediction and GT mask
	imwrite(prediction, strcat(save, basename, '.png'));
	if vessels_exist
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