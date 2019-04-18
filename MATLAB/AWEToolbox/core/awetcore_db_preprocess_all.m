function [db, annotations] = awetcore_db_preprocess_all(db, annotations)
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
    
    if awetcore_func_exists('awet_database_preprocess', 0)
        [db, annotations] = awet_database_preprocess(db, annotations);
    end
end