function results = awet_predict(model, X_test)
	% Predict results based on test set and learned model
    %
    % Input:
    %	 model   = learned model
	%	 X_test  = matrix of test data
    %
    % Output:
    %    results = array of test class values (y_test)
	
    results = zeros(size(model, 1), size(X_test, 1));
    X_test = double(X_test);
    
    for i = 1:size(model, 1)
         res = sim(model{i}, X_test');
         results(i, :) = res';
    end
end