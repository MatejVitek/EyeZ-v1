function histo = compute_rilpq_features(X,LPQfilters,charOri);
    
% X=padarray(X',2,'replicate','both');
% X=padarray(X',2,'replicate','both');
XL = ri_lpq(X,LPQfilters,charOri,'im');
His = blockproc(double(XL),[16,16], @(x) hist(x.data(:)',0:255),'BorderSize',[2,2],'TrimBorder',false);
% whos His
histo = His(:);

end


