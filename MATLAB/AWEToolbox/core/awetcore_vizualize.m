function awetcore_vizualize(targets, outputs, scores)
    global awet;
    data = struct();
    special = 1;
    max_or_thresh_mode = 3; % 1 or 2 or 3
    threshold = 0.5;
    
    targets = double(targets);
    outputs = double(outputs);
    scores = double(scores);
    
    if special > 0
        %outputs = outputs > 0.5;
        accuracies = zeros(size(outputs, 1), 1);
        
        %if max_or_thresh_mode == 3
       %    targets = vec2ind(targets)';
        %end
        
        for ri = 1:size(outputs, 1)
            output1 = squeeze(outputs(ri, :, :));
            targets1 = squeeze(targets(ri, :, :));
            
            if max_or_thresh_mode == 3
                targets1 = vec2ind(targets1)';
            end
            if max_or_thresh_mode == 1
                [~, maxI] = max(output1, [], 2);
                %positives = full(ind2vec(maxI'))';
                
                positives = zeros(size(outputs, 2), size(outputs, 3));
                for g = 1:size(positives, 1)
                    positives(g, maxI(g)) = 1;
                end
                
                err_count = sum(sum(abs(positives - targets)));
                all_count = numel(targets);
            elseif max_or_thresh_mode == 2
                positives = output1 > threshold;
                
                err_count = sum(sum(abs(positives - targets)));
                all_count = numel(targets);
                
            elseif max_or_thresh_mode == 3
                %targets = 1:(size(targets, 1));
                %output1d = zeros(size(output1, 1));
                %for d = 1:(size(output1, 1))
                %    eld = output1(d,d);
                %    if eld > 0
                %        output1d(d) = d;
                %    end
                %end
                
                positives = vec2ind(output1)';
                
                err_count = sum((positives - targets1) ~= 0);
                all_count = size(targets1, 1);                
            end
            
            accuracies(ri) = (1- (err_count / all_count)) * 100;
        end
        
        rocTar = squeeze(targets(1, :, :));
        rocSco = squeeze(scores(1, :, :));
        rocOut = squeeze(outputs(1, :, :));
        [tpr,fpr,thresholds] = roc(rocTar, rocSco);
        % confusion matrix [c,cm,ind,per] = confusion(targets,outputs)
        %plotroc(rocTar, rocSco);
        %plotconfusion(rocTar, rocOut)
        
        accuracy = mean(accuracies);
        deviation = std(accuracies);
        awetcore_log(['Accuracies are: ', num2str(accuracies'), '%%'], 0);
        awetcore_log(['Standard deviation is: ', num2str(deviation)], 0);        
    elseif special == -1
        accuracies = zeros(size(outputs, 1), 1);
        for ri = 1:size(outputs, 1)
            output1 = squeeze(outputs(ri, :, :))';
            err_count = sum((output1 - targets) ~= 0);
            all_count = size(targets, 1);
            
            accuracies(ri) = (1- (err_count / all_count)) * 100;
        end
        
        accuracy = mean(accuracies);
        deviation = std(accuracies);
        awetcore_log(['Accuracies are: ', num2str(accuracies'), '%%'], 0);
        awetcore_log(['Standard deviation is: ', num2str(deviation)], 0);        
    else
        if (isequal(awet.ident_or_verif, 0))
            err_count = sum((outputs - targets) ~= 0);
        else
            err_count = sum(abs(outputs - targets));

            TP = sum(and(outputs, targets));
            TN = sum(and(1-outputs, 1-targets));
            FP = sum(and(xor(outputs, targets), outputs));
            FN = sum(and(xor(outputs, targets), 1 - outputs));

            P = sum(targets);
            N = sum(1-targets);

            specificity = TN / (TN + FP); % TNRate TN / N
            sensitivity = TP / (TP + FN); % TPRate TP / P
            FNRate = FN / (TP + FN); % = FN / P;
            FPRate = FP / (TN + FP); % = FP / N;

            %[~,~,~,AUC] = perfcurve(targets, outputs, 1);

            data.TP = TP;
            data.TN = TN;
            data.FP = FP;
            data.FN = FN;

            data.P = P;
            data.N = N;

            data.specificity = specificity; 
            data.sensitivity = sensitivity;
            data.FNRate = FNRate;
            data.FPRate = FPRate;

            %data.AUC = AUC;
        end

        all_count = size(targets, 1);
        accuracy = (1- (err_count / all_count)) * 100;
    end    
    
    data.accuracy = accuracy;
    data.all_count = all_count;
    
    if (deviation)
        data.deviation = deviation;
    end

    % It is important that we have as high specificity as possible.
    % If we have low sensitivity this is not such a big problem.

    results_folder = strcat(awet.current_database.path, '-', ...
        awet.current_extractor.path, '-', awet.current_model.path, ...
        '-', awet.current_model.mode(1:3), '/');
    results_file = strcat(awet.run_id, '.json');

    awetcore_log(['Overall accuracy is: ', num2str(accuracy), '%%'], 0);
    awetcore_log_file([awet.current_database.path, '\t', awet.current_extractor.path, '\t', awet.current_model.path, '\t', num2str(accuracy), '\t', num2str(deviation)]);
    awetcore_write_json([awet.results_path, results_folder], results_file, data);
    awetcore_log(['Results stored into ''', awet.results_path, results_folder, ''''], 0);
end

