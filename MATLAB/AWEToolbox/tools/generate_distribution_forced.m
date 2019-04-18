function [divisions] = generate_distribution_forced(fileIn, fileOut, kfold)
    % 1. Permute classes
    % 2. Iteratively put all samples from one class until lower than limit
    %   of size/kfold
    % 3. split last one, unless in this or next group only one sample:
    %   if only one sample than put the whole group in here
    y = dlmread(fileIn);
    classes = (1:100)';
    divisions = cell(kfold, 1);

    sizeLimitTest = round((1/kfold) * numel(y(y~=0)));
    csp = randperm(size(classes, 1));
    classes = classes(csp);
    
    itest = 1;
    k = 1;
    indTest = uint32(zeros(sizeLimitTest, 1));
    
    for i = 1:size(classes, 1)
        class = classes(i);
        ex = y(class,:);
        ex = ex(ex~=0)';
        exs = size(ex, 1);
        nd = itest+exs-1;
        if (nd <= sizeLimitTest)
            indTest(itest:nd) = ex;
            itest = itest + exs;
        end
        if (i == size(classes, 1) && nd < sizeLimitTest)
            indTest(nd+1:end) = [];
        end
        if (nd == sizeLimitTest || i == size(classes, 1))
            divisions{k} = indTest';
            indTest = uint32(zeros(sizeLimitTest, 1));
            k = k + 1;
            itest = 1;
        end

        if (nd > sizeLimitTest)
            % it is more, now check if we can split it (in in both groups
            % there is at least two samples
            diffOver = nd - sizeLimitTest;
            diffUnder = exs - diffOver;
            
            if (diffOver > 1 && diffUnder > 1 && k < kfold)
                % we can split it
                indTest(itest:end) = ex(1:diffUnder);
                divisions{k} = indTest';
                
                indTest = uint32(zeros(sizeLimitTest, 1));
                indTest(1:diffOver) = ex(diffUnder+1:end);
                    
                itest = diffOver + 1;
            elseif (diffUnder > 1 || k == kfold) % the second kfold will have only one, so we put current class completely into this one
                % add the remaining part
                disp(itest);
                disp(size(indTest));
                disp(diffUnder);
                disp(size(ex));
                disp('--');
                indTest(itest:end) = ex(1:diffUnder);
                indTest = [indTest; ex(diffUnder+1:end)]; %#ok<AGROW>
                divisions{k} = indTest';
                indTest = uint32(zeros(sizeLimitTest, 1));
                itest = 1;
            else % this fold will have only one, so we put current class into the next one
                indTest(itest:end) = [];
                divisions{k} = indTest';
                indTest = uint32(zeros(sizeLimitTest, 1));
                indTest(1:exs) = ex;
                itest = exs + 1;
            end
            k = k + 1;
        end
    end
    neki = 23;

    disp(['Writing to ', '.', fileOut]);
    %       dlmwrite(fileOut, divisions, 'delimiter', ' ');
    %fileID = fopen('joppa.txt','w');
%     for x = 1:kfold
%         fprintf(fileID, '%d', divisions{x});
%     end
    %fclose(fileID);
    T = cell2table(divisions);
    writetable(T, fileOut);
end

