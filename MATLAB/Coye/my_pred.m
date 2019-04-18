function prediction = my_pred(image, sclera)
	%s = size(image);

	% Segment sclera from image
	im = segment(image, sclera);
	
	% Resize image for easier computation
	%im = imresize(im, [584 565]);
	
	% Convert RGB to Gray via PCA
	lab = rgb2lab(im);
	f = 0;
	wlab = reshape(bsxfun(@times,cat(3,1-f,f/2,f/2),lab),[],3);
	[C,S] = pca(wlab);
	S = reshape(S,size(lab));
	S = S(:,:,1);
	gray = (S-min(S(:)))./(max(S(:))-min(S(:)));
	%% Contrast Enhancment of gray image using CLAHE
	J = adapthisteq(gray,'numTiles',[8 8],'nBins',128);
	%% Background Exclusion
	% Apply Average Filter
	h = fspecial('average', [9 9]);
	JF = imfilter(J, h);
	%figure, imshow(JF)
	% Take the difference between the gray image and Average Filter
	Z = imsubtract(JF, J);
	%figure, imshow(Z)
	%% Threshold using the IsoData Method
	%level=isodata(Z); % this is our threshold level
	%level = graythresh(Z);
	%% Convert to Binary
	%BW = imbinarize(Z, level-.008);
	%% Remove small pixels
	%BW2 = bwareaopen(BW, 100);
	%% Overlay
	%BW2 = imcomplement(BW2);
	%out = imoverlay(im, BW2, [0 0 0]);
	%prediction = imresize(BW2, s(1:2));
	Z(Z<0) = 0;
	prediction = Z / max(Z(:));
	%figure, imshow(out)

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