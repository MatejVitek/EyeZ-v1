function [targets, outputs] = awetcore_evaluate(features)
    global awet;

    targets = [];
    outputs = [];
    
    %X = features.feature;
    X = features{:,2:end};
    %y = features.class;
    y = features{:,1};
    classes = unique(y);
    
    % shuffle the input
    shuffleinx = randperm(size(X, 1));
    X = X(shuffleinx, :);
    y = y(shuffleinx, :);

    if (isequal(awet.crossvalind.method, 'Kfold'))
        indices = crossvalind('Kfold', size(X, 1), awet.crossvalind.factor);
        for i = 1:awet.crossvalind.factor
            iTrain = (indices ~= i);
            iTest = (indices == i);
            X_train = X(iTrain, :);
            y_train = y(iTrain, :);
            X_test = X(iTest, :);
            y_test = y(iTest, :);
            
            [targetsTmp, outputsTmp] = awetcore_evaluate_exec(X_train, y_train, X_test, y_test, classes);
            targets = [targets; targetsTmp];
            outputs = [outputs; outputsTmp];
        end
    elseif (isequal(awet.crossvalind.method, 'HoldOut'))
        [Train, Test] = crossvalind('HoldOut', size(X, 1), awet.crossvalind.factor);
        X_train = X(Train, :);
        y_train = y(Train, :);
        X_test = X(Test, :);
        y_test = y(Test, :);
        
        [targets, outputs] = awetcore_evaluate_exec(X_train, y_train, X_test, y_test, classes);
    end
end

function [targets, outputs] = awetcore_evaluate_exec(X_train, y_train, X_test, y_test, classes)
    global awet;
    outputs = [];
    targets = [];
    if isequal(awet.ident_or_verif, 1)
        % VERIFICATION - combine all
        for i = 1:numel(classes)
            class = classes(i);
            y_train = (y_train == class);
            y_test = (y_test == class);
            awetcore_log(['True class is ', num2str(class)], 2);
            output = awet_evaluate(X_train, y_train, X_test);
            %output = awet_evaluateX(X_train, y_train, X_test, y_test);
            outputs = [outputs; output];
            targets = [targets; y_test];
        end
    else
        % IDENTIFICATION
        outputs = awet_evaluate(X_train, y_train, X_test);
        %outputs = awet_evaluateX(X_train, y_train, X_test, y_test);
        targets = y_test;
    end
end