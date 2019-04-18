function [indTrain, indTest] = awetcore_divide_db_type1(features, factor)
    y = features{:,1};    
    %shuffleinx = randperm(size(y, 1));
    %y = y(shuffleinx, :);
    classes = unique(y);
    ind = zeros(size(classes, 1), 1);
    indmax = zeros(size(classes, 1), 1);
    ind(:) = 2;
    sizeLimit = (1 - (1/factor)) * size(y, 1);
    stop = 0;
    
    while (true)
       for i = 1:size(classes,1)
           class = classes(i);
           if ind(i) < sum(y==class);
               ind(i) = ind(i) + round(rand);
           end
           indmax(i) = sum(y==class);
           %disp(sum(ind));
           if sum(ind) >= sizeLimit
               stop = 1;
               break;
           end
       end
       if stop == 1
           break;
       end
    end
    
    indTrain = zeros(sum(ind), 1);
    indTest = zeros(size(y, 1) - sum(ind), 1);
    
    m = 1;
    n = 1;
    for i = 1:size(ind,1)
        c = find(y==classes(i));
        g = randperm(size(c, 1));
        c = c(g);
        trainSize = ind(i);
        testSize = indmax(i) - ind(i);
        indTrain(m:(m+trainSize-1)) = c(1:ind(i));
        indTest(n:(n+testSize-1)) = c((ind(i)+1):end);
        %disp([num2str(indmax(i)), ' vs ', num2str(size(c, 1))]);
        m = m + trainSize;
        n = n + testSize;
    end
end

