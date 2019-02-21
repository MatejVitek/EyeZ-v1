% This computes LBP features from the image
function histo = compute_lbp_features(X,mapping,block_size)
    
X=padarray(X',2,'replicate','both');
X=padarray(X',2,'replicate','both');
XL=lbp_SAM(X,2,8,mapping,'i');

%XR = fliplr(XL);
% His = blockproc(double(XL),[6,6], @(x) hist(x.data(:)',0:58),'BorderSize',[0,0],'TrimBorder',false);
His = blockproc(double(XL),[8,8], @(x) hist(x.data(:)',0:58),'BorderSize',[4,4],'TrimBorder',false);
% whos His
histo = His(:);

end