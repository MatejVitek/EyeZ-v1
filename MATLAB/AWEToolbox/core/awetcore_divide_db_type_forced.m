function [divisions] = awetcore_divide_db_type_forced(y, kfold)
    % 1. Permute classes
    % 2. Iteratively put all samples from one class until lower than limit
    %   of size/kfold
    % 3. split last one, unless in this or next group only one sample:
    %   if only one sample than put the whole group in here
    
    divisions = cell(kfold, 1);
    classes = uint32(unique(y));
    sizeLimitTest = round((1/kfold) * size(y, 1));
    csp = randperm(size(classes, 1));
    classes = classes(csp);
    
    itest = 1;
    k = 1;
    indTest = uint32(zeros(sizeLimitTest, 1));
    
    for i = 1:size(classes, 1)
        isLastOne = (i == size(classes, 1));
        class = classes(i);
        ex = find(y==class);
        exs = size(ex, 1);
        nd = itest+exs-1;
        if (nd <= sizeLimitTest)
            indTest(itest:nd) = ex;
            itest = itest + exs;
        end
        if (isLastOne && nd < sizeLimitTest)
            indTest(nd+1:end) = [];
        end

        if (nd > sizeLimitTest)
            % it is more, now check if we can split it (in in both groups
            % there is at least two samples
            diffOver = nd - sizeLimitTest;
            diffUnder = exs - diffOver;
            
            if (k == kfold) % just add to current
                indTest(itest:end) = ex(1:diffUnder);
                indTest = [indTest; ex(diffUnder+1:end)]; %#ok<AGROW>
                divisions{k} = indTest;

                itest = itest + exs;
            elseif (diffOver > 1 && diffUnder > 1 && k < kfold)
                % we can split it
                indTest(itest:end) = ex(1:diffUnder);
                divisions{k} = indTest;
                
                indTest = uint32(zeros(sizeLimitTest, 1));
                indTest(1:diffOver) = ex(diffUnder+1:end);
                    
                itest = diffOver + 1;
                 
                k = k + 1;
            elseif (diffUnder > 1) % the second kfold will have only one, so we put current class completely into this one
                % add the remaining part
                indTest(itest:end) = ex(1:diffUnder);
                indTest = [indTest; ex(diffUnder+1:end)]; %#ok<AGROW>
                divisions{k} = indTest;
                indTest = uint32(zeros(sizeLimitTest, 1));
                itest = 1;
                
                k = k + 1;
            else % this fold will have only one, so we put current class into the next one
                indTest(itest:end) = [];
                divisions{k} = indTest;
                indTest = uint32(zeros(sizeLimitTest, 1));
                indTest(1:exs) = ex;
                itest = exs + 1;
            
                k = k + 1;
            end
            
        end
        
        if (nd == sizeLimitTest || isLastOne)
            divisions{k} = indTest;
            indTest = uint32(zeros(sizeLimitTest, 1));
            k = k + 1;
            itest = 1;
        end
    end
end

