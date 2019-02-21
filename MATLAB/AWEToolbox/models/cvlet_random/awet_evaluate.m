function results = awet_evaluate(~, y_train, X_test)
	% Predict results based on train and test set
    %
    % Input:
    %    X_train = matrix of train data
    %    y_train = array of train class values
	%	 X_test  = matrix of test data
    %
    % Output:
    %    results = array of test class values (y_test)
	
    classes = unique(y_train);
    classCount = numel(classes);
    sampleSize = size(X_test, 1);
    if (classCount <= 2)
        awetcore_log('BINARY mode', 2);
        results = round(rand(sampleSize,1));
    else
        awetcore_log('MULTI mode', 2);
        results = datasample(classes, sampleSize);
    end
end