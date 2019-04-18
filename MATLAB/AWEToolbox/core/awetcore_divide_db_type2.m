function [indTrain, indTest] = awetcore_divide_db_type2(y, factor)
    % 1. Randomly put items in two arrays
    % 2. When you come to the end, split the last class in two if needed,
    % if not we're OK
    % 3. If the last class has at least two items in each group we're ok
    %    If not, repeat the process 
      
    classes = uint32(unique(y));
    indmax = zeros(size(classes, 1), 1);
    sizeLimitTrain = round((1 - (1/factor)) * size(y, 1));
    sizeLimitTest = round((1/factor) * size(y, 1));
    indTrain = uint32(zeros(sizeLimitTrain, 1));
    indTest = uint32(zeros(sizeLimitTest, 1));
    
    for i = 1:size(classes,1)
        class = classes(i);
        indmax(i) = sum(y==class);
    end
    
    % Walk through classes and put them into test:
    % TODO
    % when totally at the sum, finish, we're OK
    % when over the sum, split the last class. If at least two in both test
    % and train, we're OK
    % if still here, start swapping items between test and train
    sentinelLimit = 5000;
    sentinel = 0;
    MINIMUM_PER_CLASS = 2;
    MINIMUM_OF_CLASSES = 3;
    while (sentinel < sentinelLimit)
        csp = randperm(size(classes, 1));
        classes = classes(csp);
        indmax = indmax(csp);
        itest = 1;
        for i = 1:size(classes, 1)
            class = classes(i);
            indmax_test = indmax(1:i);
            indmax_train = indmax((i+1):end);
            sum_test = sum(indmax_test);
            sum_train = sum(indmax_train);
            status = -1; % we went over, examples will need to be split between

            if (sum_test == sizeLimitTest)
                % everything is fine and we're done, break the loop
                status = 1;
            elseif (sum_test < sizeLimitTest)
                % everything is fine for now, continue with the loop
                status = 0;
            end            

            if status >= 0
                % assign the whole class into one group
                ex = find(y==class);
                exs = size(ex, 1);
                indTest(itest:(itest+exs-1)) = ex;
                itest = itest + exs;
            end

            if status == 1 || status == -1
                % because we have come to the end, assign all the remaining
                % values to the train set
                itrain = 1;
                for j = (i+1):size(classes, 1)
                    class_train = classes(j);
                    ex = find(y==class_train);
                    exs = size(ex, 1);
                    indTrain(itrain:(itrain+exs-1)) = ex;
                    itrain = itrain + exs;
                end
            end

            if status == -1
                % split the last class in to two parts
                ex = find(y==class);
                exs = size(ex, 1);

                diff_test = exs - sum_test + sizeLimitTest;
                diff_train = sizeLimitTrain - sum_train;
                
                if (diff_test >= MINIMUM_PER_CLASS && diff_train >= MINIMUM_PER_CLASS)
                    status = 1;
                    % split in two and we're done
                    indTest(end-diff_test+1:end) = ex(1:diff_test);
                    indTrain(end-diff_train+1:end) = ex(diff_test+1:end);
                elseif diff_test < MINIMUM_PER_CLASS || diff_train < MINIMUM_PER_CLASS
                    % re-run the whole process ...
                    status = -2;
                end
            end

            if status == 1 || status == -2
                break;
            end
        end

        if status == 1
            if (size(unique(y(indTrain)), 1) >= MINIMUM_OF_CLASSES && size(unique(y(indTest)), 1) >= MINIMUM_OF_CLASSES)
                break;
            end
        else
            %awetcore_log(['Random distrubution re-run, status was: ', num2str(status), '\n'], 2);
        end
        sentinel = sentinel + 1;
    end
    awetcore_log(['Random distrubution completed in ', num2str(sentinel+1), ' trys.'], 2);
    if (sentinel == sentinelLimit)
        awetcore_log('FATAL ERROR STOPPING TOOLBOX EXECUTION: was unable to generate sets from the DB', 0);
        return
    end;
end

