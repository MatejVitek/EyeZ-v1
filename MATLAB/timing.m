%% Config
% Source directory for images
src_dir = '/hdd/EyeZ/SBVPI/Subsets/Rot ScleraNet/stage2_ungrouped/';

% Matching algorithm
%alg_dir = 'Miura';
alg_dir = 'Coye';
%alg_dir = 'B-COSFIRE';
old_dir = cd(alg_dir);

% Directory to save results to
save_dir = strcat('/hdd/EyeZ/Segmentation/Results/Vessels/', alg_dir);
overwrite = false;

%for alg_name = {'MC', 'RLT'}
%	time_alg_pair(alg_name{1}, src_dir, 100);
%end

time_alg(@my_pred, src_dir, 10);

cd(old_dir);

%% Evaluate binarised and normalised algorithm version
function time_alg_pair(alg_name, src_dir, n)
	alg = @(image, sclera)(my_pred(image, sclera, alg_name));
	time_alg(alg, src_dir, n);

	alg = @(image, sclera)(my_pred(image, sclera, alg_name, 'norm'));
	time_alg(alg, src_dir, n);
end

%% Evaluate a single algorithm
function time_alg(alg, src_dir, n)
	set = imageSet(src_dir, 'recursive');
	if isempty(gcp('nocreate'))
		parpool(24);
	end
	timer = tic;

	n_img = 0;
	for id_image = 1:set.Count
		if n_img > n
			break
		end
		% If it's a mask image, skip it
		[path, basename, ~] = fileparts(set.ImageLocation{id_image});
		if regexp(basename, '\d+[LR]_[lrsu]_\d+_')
			continue
		end

		% Evaluate a single image
		score = evaluate(path, basename, alg);

		% If a mask was not found, skip
		if score == -1
			continue
		end
		
		disp(basename);
		n_img = n_img + 1;
	end
	disp('Time per image:');
	disp(toc(timer) / n_img);
end

%% Load an image and its corresponding sclera prediction and vessels GT mask and evaluate the algorithm on it
function score = evaluate(path, basename, alg)
	sclera_file = strcat(path, '/', basename, '_sclera.png');
	
	if ~isfile(sclera_file)
		score = -1;
		return
	end

	% Load base image
	image = im2double(imread(strcat(path, '/', basename, '.jpg')));
	s = size(image);

	% Load masks
	sclera = im2double(rgb2gray(imread(sclera_file)));

	% Run the matching algorithm
	alg(image, sclera);

	score = 0;
end
