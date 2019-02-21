function [db, annotations] = awetcore_preprocess_all(db, annotations)
    % preprocess applies preprocessing to each image in the db
    % 
    % Input:
    %    db          = all ear images
    %    annotations = annotation data for
    % 			        each image in db
    % 
    % Output:
    %    db          = modified images
    %    annotations = modified annotations
    
    if awetcore_func_exists('awet_preprocess', 0)
        if size(annotations, 1) == 0
            annotations = zeros(size(db,1), 0);
        end
        for i = 1:size(db,1)
            I = db(i, :).image{1};
            if (size(annotations, 1) > 0)
                annotations_row = annotations(i, :);
            else
                annotations_row = [];
            end
            [db(i, :).image{1}, annotations(i, :)] = awet_preprocess(I, annotations_row);
        end
    end
end