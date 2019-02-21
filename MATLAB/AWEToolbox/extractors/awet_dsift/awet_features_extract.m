function [features] = awet_features_extract(I, I_annotations)
    % extract ear features for current image
    %
    % Input:
    %    I             = already preprocessed ear image
    %    I_annotations = annotation data for this image
    %
    % Output:
    %    features      = vector of ear features
    
    %[f, features] = vl_dsift(I, 'step', 10, 'size', 4, 'geometry', [4 4 8]);
    
    binSize = 8;
    magnif = 16;
    f = zeros(4, 100);
    i = 1;
    for y = ((1:10:100) + 4)
        for x = ((1:10:100) + 4)
            f(1, i) = x;
            f(2, i) = y;
            i = i + 1;
        end
    end
 
    f(3,:) = binSize/magnif ;
    f(4,:) = 0 ;
    [~, features] = vl_sift(I, 'frames', f) ;
 end