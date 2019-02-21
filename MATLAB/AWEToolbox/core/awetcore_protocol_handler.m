function [features_divided, classes] = awetcore_protocol_handler(features, divisions)
% It takes distributions from divisions variable
    global awet;

    X = double(features{:,2:end});
    y = uint32(features{:,1});
    classes = unique(y);
    len = size(divisions, 1); % needs to be the same as k in k-fold

    features_divided = struct();
    features_divided(len).X_test = [];
    features_divided(len).y_test = [];

    for i = 1:len
        i_test = divisions{i};
        features_divided(i).X_test = double(X(i_test, :));
        tmp = uint32(y(i_test, :));
        features_divided(i).y_test = tmp;
    end
end