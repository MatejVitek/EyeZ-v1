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
	
    classCount = numel(unique(y_train));
    kernelFunction = 'linear';
    if (classCount <= 2)
        % CSVM
        SVMModel = fitcsvm(X_train, y_train);
        [results, ~] = predict(SVMModel, X_test); % ~ = score
    else
        % MCSVM
        SVMTemplate = templateSVM('KernelFunction', kernelFunction);
        SVMModel = fitcecoc(X_train, y_train, 'Learners', SVMTemplate);
        [results, ~] = predict(SVMModel, X_test); % ~ = score
    end
end

function [result] = multisvm(TrainingSet,GroupTrain,TestSet)
    %Models a given training set with a corresponding group vector and 
    %classifies a given test set using an SVM classifier according to a 
    %one vs. all relation. 
    %
    %This code was written by Cody Neuburger cneuburg@fau.edu
    %Florida Atlantic University, Florida USA
    %This code was adapted and cleaned from Anand Mishra's multisvm function
    %found at http://www.mathworks.com/matlabcentral/fileexchange/33170-multi-class-support-vector-machine/

    u=unique(GroupTrain);
    numClasses=length(u);
    result = zeros(length(TestSet(:,1)),1);

    %build models
    for k=1:numClasses
        %Vectorized statement that binarizes Group
        %where 1 is the current class and 0 is all other classes
        G1vAll=(GroupTrain==u(k));
        options.MaxIter = 1000000;
        
        models{k} = fitcsvm(TrainingSet, G1vAll);
    end

    %classify test cases
    for j=1:size(TestSet,1)
        for k=1:numClasses
            if(predict(models{k}, TestSet(j,:))) 
                break;
            end
        end
        result(j) = k;
    end
end