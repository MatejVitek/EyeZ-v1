function [indTrain, indTest] = awetcore_divide_db_type2off(features, factor)
    if (~exist('distribute_classes','file'))        
        addpath(genpath('libraries/knapsackori'));
    end

    y = features{:,1};    
    classes = unique(y);
    indmax = zeros(size(classes, 1), 1);
    sizeLimitTrain = round((1 - (1/factor)) * size(y, 1));
    sizeLimitTest = round((1/factor) * size(y, 1));
    indTrain = zeros(sizeLimitTrain, 1);
    indTest = zeros(sizeLimitTest, 1);
    
    for i = 1:size(classes,1)
        class = classes(i);
        indmax(i) = sum(y==class);
    end
    
    [distribution_test, opt_val] = distribute_classes(indmax, sizeLimitTest);
    distribution_train = ~distribution_test;
    
    %classes_train = classes .* distribution_train;
    %classes_test = classes .* distribution_test;
    m = 1;
    n = 1;
    for i = 1:size(classes,1)
        class = classes(i);
        is_train = distribution_train(i);
        is_test = distribution_test(i);
        if (is_train && is_test || ~is_train && ~is_test)
            disp('FATAL ERROR!!!');
        end
        x = find(y==class);
        xs = size(x, 1);
        xrp = randperm(xs);
        if is_train
            indTrain(m:m+xs-1) = x(xrp);
            m = m + xs;
        else
            indTest(n:n+xs-1) = x(xrp);
            n = n + xs;
        end
    end
    
    
end

