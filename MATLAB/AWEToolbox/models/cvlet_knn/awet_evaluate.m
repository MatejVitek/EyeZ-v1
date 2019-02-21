function results = awet_evaluate(X_train, y_train, X_test)
	% Predict results based on train and test set
    %
    % Input:
    %    X_train = matrix of train data
    %    y_train = array of train class values
	%	 X_test  = matrix of test data
    %
    % Output:
    %    results = array of test class values (y_test)
	
    %chiSqrDist = @(x,Z)sqrt((bsxfun(@minus,x,Z).^2));
	%SVMModel = fitcknn(X_train, y_train, 'Distance', @(x,Z)chiSqrDist(x,Z), 'Standardize', 1);
    X_train = double(X_train);
    X_test = double(X_test);
    SVMModel = fitcknn(X_train, y_train);
	[results, ~] = predict(SVMModel, X_test);
end