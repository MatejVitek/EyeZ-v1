function [distancesF, mMin, mMax] = awetcore_evaluate_constant(features_divided, all_classes, distance_method)
    global awet;
    
    distances = struct();
    len = size(features_divided, 2);
    
    distances(len).output = [];
    distances(len).target = [];
    distances(len).features_len = 0;
    
    matMinAll = realmax;
    matMaxAll = realmin;

    for i = 1:len
        %%
        % for each fold calculate matrix (ROC etc. will be calculated in
        % the final visulization function

        features = double(features_divided(i).X_test);
        classes = features_divided(i).y_test;
        distances(i).features_len = size(features, 2);

        % Norm features:   
        if (awet.norm_features)  
            features2 = bsxfun(@rdivide, features, sum(abs(features), 2));
            features2(features==0) = 0;
            features = features2;
        end

        % Calc distances
        if (strcmp(distance_method, 'chi') == 1)
            %awetcore_log(['Calculating Chi-square distance matrix ', num2str(i), '/', num2str(len)], 2);
            distance = pdist2(features, features, @chi_square_statistics_fast);
        else
            distance = pdist2(features, features, distance_method);
        end
        distance(isnan(distance)) = 1;
        
        matMin = min(distance(:));
        matMax = max(distance(:));
        if (matMin < matMinAll)
            matMinAll = matMin;
        end
        if (matMax > matMaxAll)
            matMaxAll = matMax;
        end

        clen = size(classes, 1);
        target = zeros(clen, clen);

        for y = 1:clen
            for x = 1:clen
                target(y, x) = classes(y) == classes(x);
            end
        end

        distances(i).classes = classes;
        distances(i).output2D = distance;
        distances(i).target2D = target;
    end
    
    mMin = matMinAll;
    mMax = matMaxAll;
    
    matMinAll = realmax;
    matMaxAll = realmin;
    
    for i = 1:len
        %% do normalization of distances
        classes = distances(i).classes;
        target2D = distances(i).target2D;
        output2D = distances(i).output2D;
        
        clients_for_hist = [];
        impostors_for_hist = [];
        
        %% do transformation to one-example per class matrix
        if (awet.compressed_evaluation == 1)
            % Transform matrices so that in each group of same classes
            % only the one with max value stays.
            % in a list of all combinations of values for one class select the
            % largest one (but, of course ignore the diagonal)
            
            [output2D, target2D, classes_x, classes_y] = awetcore_distances_to_one_class(output2D, target2D, classes);
            
            matMin = min(output2D(:));
            matMax = max(output2D(:));
            if (matMin < matMinAll)
                matMinAll = matMin;
            end
            if (matMax > matMaxAll)
                matMaxAll = matMax;
            end
            
            distances(i).output2D = output2D;
            distances(i).target2D = target2D;
            distances(i).classes_x = classes_x;
            distances(i).classes_y = classes_y;
        else
            distances(i).classes_x = classes;
            distances(i).classes_y = classes;
        end
        
        %% store 1D version
        if (0 && awet.compressed_evaluation == 0)
            distances(i).output = output2D(triu(true(size(output2D)), 1));
            distances(i).target = target2D(triu(true(size(target2D)), 1));
        else
            distances(i).output = output2D(:);
            distances(i).target = target2D(:);
        end
%         disp(size(output2D));
%         disp(size(target2D));
%         disp('---');
        clients_for_hist = [clients_for_hist; output2D(logical(target2D))];
        impostors_for_hist = [impostors_for_hist; output2D(~logical(target2D))];
    end
    
    mMin = matMinAll;
    mMax = matMaxAll;
    
    if (awet.current_database.protocol == 3)
        bootstrp = 100;
        lenBoot = bootstrp * len;
        
        distancesF = struct();
        distancesF(lenBoot).output2D = [];
        distancesF(lenBoot).target2D = [];
        distancesF(lenBoot).output = [];
        distancesF(lenBoot).target = [];
        distancesF(lenBoot).classes = [];
        distancesF(lenBoot).features_len = 0;
        distancesF(lenBoot).classes_x = [];
        distancesF(lenBoot).classes_y = [];
        
        step = 1;
        for i = 1:len
            nxtStep = step + bootstrp;
            neki = perform_bootstrap(distances(i), bootstrp, 0.60);
            distancesF(step:nxtStep-1) = neki;
            step = nxtStep;
        end
    else
        distancesF = distances;
    end
    

    awetcore_plot_hist(clients_for_hist, impostors_for_hist);
end

function distancesF = perform_bootstrap(distances, iterations, ratio)
    global awet;
    
    classes = distances.classes;
    classes_x = distances.classes_x;
    classes_y = distances.classes_y;
    target2D = distances.target2D;
    output2D = distances.output2D;
    
    distancesF = struct();
    distancesF(iterations).output2D = [];
    distancesF(iterations).target2D = [];
    distancesF(iterations).output = [];
    distancesF(iterations).target = [];
    distancesF(iterations).classes = [];
    distancesF(iterations).features_len = 0;
    
    [~, booti] = bootstrp(iterations, @(~,~) 0, 1:round(size(classes, 1) * ratio));

    for i = 1:size(booti, 2)
        boot = booti(:, i);
        perm = randperm(length(classes_y));
        classes = classes(perm);
        classes_y = classes_y(perm);
        output2D = output2D(perm, :);
        target2D = target2D(perm, :);
        
        distancesF(i).classes = classes(boot);
        distancesF(i).classes_x = classes_x;
        distancesF(i).classes_y = classes_y(boot);
        distancesF(i).output2D = output2D(boot, :);
        distancesF(i).target2D = target2D(boot, :);
        
        %% store 1D version
        if (awet.compressed_evaluation == 0)
            distancesF(i).output = output2D(triu(true(size(output2D)), 1));
            distancesF(i).target = target2D(triu(true(size(target2D)), 1));
        else
            distancesF(i).output = distancesF(i).output2D(:);
            distancesF(i).target = distancesF(i).target2D(:);
        end
    end
end