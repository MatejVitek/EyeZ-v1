function [train, test] = generate_distribution(classes, examples_per_class, train_percent, folder)
    disp([classes, examples_per_class]);
    R = zeros(classes, examples_per_class);
    offset = 0;
    for i = 1:classes
        line = randperm(examples_per_class);
        R(i,:) = line + offset;
        offset = offset + examples_per_class;
    end
    
    delimiter = round(train_percent * examples_per_class);
    train = R(:, 1:delimiter);
    %trainr = randperm(numel(train))';
    test = R(:, (delimiter+1):end);
    %testr = randperm(numel(test))';
    
    classes_list = (1:classes)';
    
    % Write into file
    if ~exist('folder', 'var')
        folder = './';
    end
    
    trainF = 'train.txt';
    %trainFr = 'train_rand.txt';
    testF = 'test.txt';
    %testFr = 'test_rand.txt';
    classesF = 'classes.txt';
    
    disp(['Writing to ', folder, trainF]);
    dlmwrite([folder, trainF], train, 'delimiter', ' ');
    %disp(['Writing to ', folder, trainFr]);
    %dlmwrite([folder, trainFr], trainr, 'delimiter', ' ');
    disp(['Writing to ', folder, testF]);
    dlmwrite([folder, testF], test, 'delimiter', ' ');
    %disp(['Writing to ', folder, testFr]);
    %dlmwrite([folder, testFr], testr, 'delimiter', ' ');
    disp(['Writing to ', folder, classesF]);
    dlmwrite([folder, classesF], classes_list, 'delimiter', ' ');
end