%% Config
global src_dir segnet_dir save_dir alg

% Source directory for images
src_dir = '/hdd/EyeZ/Rot/Resized SBVP/SBVP_vessels_480x360/';

% Source directory for SegNet sclera mask predictions
segnet_dir = '/hdd/EyeZ/Rot/Segmentation/Results/segnet/';

% Matching algorithm
alg_dir = 'Miura';
%alg_name = 'MC';
%alg_name = 'RLT';
alg_name = 'RLTGS';
addpath(strcat('./', alg_dir));
alg = @(image, sclera)(my_pred(image, sclera, alg_name));
%alg = @(image, sclera)(my_pred(image, sclera, alg_name, 'norm'));

% Directory to save results to
save_dir = strcat('/hdd/EyeZ/Rot/Segmentation/Results/', alg_dir, '_', alg_name, '/');
%save_dir = strcat('/hdd/EyeZ/Rot/Segmentation/Results/', alg_dir, '_', alg_name, '_norm/');
if ~exist(save_dir, 'dir')
	mkdir(save_dir);
end

%% Evaluate matching algorithm on the dataset
set = imageSet(src_dir, 'recursive');
%results = zeros(1, sum(vertcat(set.Count)));
k = 0;
for id_images = set
	% Iterate over all images of current ID
	for image = id_images.ImageLocation
		% If it's a mask image, skip it
		[path, basename, ~] = fileparts(image{1});
		if regexp(basename, '\d+[LR]_[lrsu]_\d+_')
			continue
		end
		
		% Evaluate a single image
		disp(basename);
		score = evaluate(path, basename);
		
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

%% Load an image and its corresponding sclera prediction and vessels GT mask and evaluate the matching algorithm on it
function score = evaluate(path, basename)
	global segnet_dir save_dir alg
	
	sclera_file = strcat(segnet_dir, basename, '.png');
	vessels_file = strcat(path, '/', basename, '_vessels.png');
	
	if ~isfile(sclera_file)
		disp('404: Sclera not found');
		score = -1;
		return
	end
	vessels_exist = isfile(vessels_file);
	
	% Load base image
	image = im2double(imread(strcat(path, '/', basename, '.JPG')));
	
	% Load masks
	sclera = im2double(rgb2gray(imread(sclera_file)));
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
	imshow(prediction);
	
	% Save prediction and GT mask
	imwrite(prediction, strcat(save_dir, basename, '.png'));
	if vessels_exist
		imwrite(vessels, strcat(save_dir, basename, '_gt.png'));
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