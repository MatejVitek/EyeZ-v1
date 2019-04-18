function features_db = awetcore_features_extract_all(db, annotations)
    % extract ear features for the current image
    %
    % Input:
    %    db          = already preprocessed ear images
    %    annotations = annotation data for all images
    %
    % Output:
    %    features    = vector of ear features for each image
    
    global awet;
    
    if awetcore_func_exists('awet_features_extract', 1)
        features = [];
        call_post_process = awetcore_func_exists('awet_postprocess', 0);
        
        ttt = [];
        clsss = zeros(size(db,1),2);
        clsss(:,1) = 1:size(clsss,1);
        for i = 1:size(db,1)
            if (size(annotations, 1) >= i)
                features_i = awet_features_extract(db(i, :).image{1}, annotations(i, :));
            else
                features_i = awet_features_extract(db(i, :).image{1}, []);
            end
            
            %% TODO make configurable
            % store features to separate file and also add to the whole file
%             tmpfld = [awet.calculated_features_path, awet.current_database.path, '/', lower(awet.current_extractor.name), '/'];
%             if ~exist(tmpfld, 'dir')
%                 mkdir(tmpfld);
%             end
%             nsift_file_path = [tmpfld, num2str(i), '.n', lower(awet.current_extractor.name)];
%             dlmwrite(nsift_file_path, features_i', 'delimiter', ' ');
            clsss(i,2) = db(i, :).class;
            sizeFeat1 = size(features_i, 1);%128
            sizeFeat2 = size(features_i, 2);%100
            sizeMult = i * sizeFeat2;
            if (i == 1)
                ttt = zeros(size(db,1) * sizeFeat2, sizeFeat1); 
            end
            ttt((sizeMult - sizeFeat2+1):sizeMult, :) = features_i';
            
            %%
            if (~awet.bulk_postprocess && call_post_process) % otherwise it is executed in awet_run.m
                features_i = awetcore_postprocess_all(features_i);
            end
            % NORM:
            %features_i = bsxfun(@rdivide, features, sum(features, 2));
            %features_i = double(features_i)/sum(features_i);
                
            %disp([num2str(i), ' of ', num2str(size(db,1)), ': ', num2str(size(features_i, 1)), 'x', num2str(size(features_i, 2))]);
            features = [features; [double(db.class(i)), double(features_i)]];
        end
        
        %features_db = table(features, db.class, 'VariableNames', {'feature' 'class'});
        featuresCount = size(features_i, 2);
        labels = strsplit(sprintf('v%i,', 1:featuresCount), ',');
        features_db = array2table(features, 'VariableNames', ['class', labels(1:end-1)]);
        
        %% TODO make configurable
        % write ttt to arff file
        if ~exist(awet.calculated_features_path, 'dir')
            awetcore_log(['Folder ''', awet.calculated_features_path, ''' does not exist, creating it ...'], 2);
            mkdir(awet.calculated_features_path)
        end
        addpath('libraries/arffwrite');
        arffFile = [awet.calculated_features_path, awet.current_database.path, '-', lower(awet.current_extractor.name), awet.parameter_path];
        arffwrite(arffFile, ttt, lower(awet.current_extractor.name));
        
        % write groundtruths
        gtruths = [awet.calculated_features_path, awet.current_database.path, '.ground_truths'];
        dlmwrite(gtruths, clsss, 'delimiter', ';');
    end
end