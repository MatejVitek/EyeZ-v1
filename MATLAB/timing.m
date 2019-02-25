%% Config
% Source directory for images
src_dir = '/hdd/EyeZ/Rot/SBVPI/SBVP_vessels/';

% Source directory for SegNet sclera mask predictions
sclera_dir = '/hdd/EyeZ/Rot/Segmentation/Results/Sclera/SegNet/';

% Matching algorithm
alg_dir = 'Miura';
addpath(strcat('./', alg_dir));

% Evaluate algorithms
for alg_name = {'MC', 'RLT'}
	time_alg_pair(alg_name{1}, src_dir, sclera_dir, 20);
end

%% Evaluate binarised and normalised algorithm version
function time_alg_pair(alg_name, src_dir, sclera_dir, n)
	alg = @(image, sclera)(my_pred(image, sclera, alg_name));
	time_alg(alg, src_dir, sclera_dir, n);
	
	alg = @(image, sclera)(my_pred(image, sclera, alg_name, 'norm'));
	time_alg(alg, src_dir, sclera_dir, n);
end

%% Evaluate a single algorithm
function time_alg(alg, src_dir, sclera_dir, n)
	set = imageSet(src_dir, 'recursive');
	k = 0;
	timer = tic;
	
	for id_images = 1:length(set)
		% Iterate over all images of current ID
		for image = set(id_images).ImageLocation
			% If it's a mask image, skip it
			[path, basename, ~] = fileparts(image{1});
			if regexp(basename, '\d+[LR]_[lrsu]_\d+_')
				continue
			end

			% Evaluate a single image
			score = evaluate(path, basename, alg, sclera_dir);

			% If a mask was not found, skip
			if score == -1
				continue
			end

			% Otherwise save the result
			k = k + 1;
			if k >= n
				disp('Time per image:');
				disp(toc(timer) / n);
				return;
			end
		end
	end
end

%% Load an image and its corresponding sclera prediction and vessels GT mask and evaluate the algorithm on it
function score = evaluate(path, basename, alg, sclera_dir)
	sclera_file = strcat(sclera_dir, basename, '.png');
	
	if ~isfile(sclera_file)
		score = -1;
		return
	end
	
	% Load base image
	image = im2double(imread(strcat(path, '/', basename, '.JPG')));
	s = size(image);
	
	% Load masks
	sclera = imresize(im2double(rgb2gray(imread(sclera_file))), s(1:2));
	
	% Run the matching algorithm
	alg(image, sclera);

	% TODO: Calculate and return error rate
	score = 0;
end