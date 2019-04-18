function model = awet_fit(X_train, Y_train)
	% Learn model and return learned model
    %
    % Input:
    %    X_train = matrix of train data
    %    y_train = array of train class values
    %
    % Output:
    %    model = learned model
	
    %chiSqrDist = @(x,Z)sqrt((bsxfun(@minus,x,Z).^2));
	%SVMModel = fitcknn(X_train, y_train, 'Distance', @(x,Z)chiSqrDist(x,Z), 'Standardize', 1);
    
    model = cell(size(Y_train, 1), 1);
    X_train = double(X_train);
    for i = 1:size(Y_train, 1)
        y_train = Y_train(i, :)';
		net = patternnet(20);
		net = train(net, X_train', logical(y_train)');
        model{i} = net;
    end
end