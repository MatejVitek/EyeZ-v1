function [features] = awet_postprocess(features)
    % do postprocessing on all features
    % 
    % Input:
    %    features = array of ear features
    %               for each image in db
    % 
    % Output:
    %    features = modified features
    % 
    % If you are doing bulk postprocess, this is the loop example:
    % for i = 1:size(features,1)
    %   [features(i, :).feature] = do_some_postprocessing(features(i, :).feature);
    % end
 
%     if (size(features) > 1)
%         featuresT = vl_kmeans(double(features), 2);
%         features = vl_kmeans(double(featuresT), 1);
%     elseif (size(features) > 0)
%        features = vl_kmeans(double(features), 1);
%     else
%         features = zeros(128, 1);
%     end
    features = transpose(reshape(features,[],1));
 end