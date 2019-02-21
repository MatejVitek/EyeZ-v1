X = imread('sample_image.bmp');
[R,L] = single_scale_retinex(X);
figure, imshow(X,[]);
figure, imshow(R,[]);
figure, imshow(L,[]);
