%% Config
% Source directory for images
src_dir = '/hdd/EyeZ/Rot/SBVPI/stage2/';

% Matching algorithm
alg_dir = 'Miura';
addpath(strcat('./', alg_dir));

% Evaluate algorithms
for alg_name = {'MC', 'RLT'}
	time_alg_pair(alg_name{1}, src_dir, 100);
end

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

	%parfor (id_image = 1:min(set.Count, n), 24)
	for id_image = 1:min(set.Count, n)
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
	end
	disp('Time per image:');
	disp(toc(timer) / min(set.Count, n));
end

%% Load an image and its corresponding sclera prediction and vessels GT mask and evaluate the algorithm on it
function score = evaluate(path, basename, alg)
	sclera_file = strcat(path, '/', basename, '_sclera.png');

	if ~isfile(sclera_file)
		score = -1;
		return
	end

	% Load base image
	image = im2double(imread(strcat(path, '/', basename, '.JPG')));
	s = size(image);

	% Load masks
	sclera = im2double(imread(sclera_file));
	eps = 0.05;
	r = sclera(:, :, 1);
	g = sclera(:, :, 2);
	b = sclera(:, :, 3);
	sclera = r < eps & g > 1 - eps & b < eps;

	% Run the matching algorithm
	alg(image, sclera);

	% TODO: Calculate and return error rate
	score = 0;
end
