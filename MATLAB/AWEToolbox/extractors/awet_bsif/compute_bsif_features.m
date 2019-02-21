function histo = compute_bsif_features(X,ICAtextureFilters)
    
X=padarray(X',2,'replicate','both');
X=padarray(X',2,'replicate','both');
XL = bsif(X,ICAtextureFilters,'im');
His = blockproc(double(XL),[18,18], @(x) hist(x.data(:)',0:255),'BorderSize',[2,2],'TrimBorder',false);
% whos His
histo = His(:);

end


