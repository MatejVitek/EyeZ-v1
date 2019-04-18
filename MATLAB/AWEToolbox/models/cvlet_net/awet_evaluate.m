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
	
%     X_train = transpose(X_train);
%     y_train = transpose(y_train);
%     X_test = transpose(X_test);
%     
%     X_train_g = nndata2gpu(X_train);
%     y_train_g = nndata2gpu(y_train);
%     X_test_g = nndata2gpu(X_test);
%     
%     net1 = feedforwardnet(1000);
%     net2 = configure(net1, X_train, y_train);
% 	net2 = train(net2, X_train_g, y_train_g, 'useGPU', 'yes','showResources','yes');
% 	y_g = net2(X_test_g,'useGPU','yes','showResources','yes');
%     y = gpu2nndata(y_g);
% 	results = transpose(y);

    classNum = numel(unique(y_train));
    X_train = transpose(X_train);
    y_train_tmp = transpose(y_train);
    X_test = transpose(X_test);
    
    y_train = zeros(classNum, size(y_train_tmp, 2));
    
    for i = 1:size(y_train, 2)
        y_train(y_train_tmp(1, i), 1) = 1;
    end
    
    net1 = feedforwardnet(1000);
	net2 = train(net1, X_train, y_train, 'useGPU', 'yes','showResources','yes');
	y = net2(X_test,'useGPU','yes','showResources','yes');
	y = transpose(y);
    results = zeros(size(y, 1), 1);
    for i = 1:size(y, 1)
        [~, maxInd] = min(y(i, :));
        results(i) = maxInd;
    end
    disp(results);
end