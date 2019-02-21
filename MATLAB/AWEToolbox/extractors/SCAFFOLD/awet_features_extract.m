function [features] = awet_features_extract(I, I_annotations)
    %% extract ear features for current image
    %%
    %% Input:
    %%    I             = already preprocessed ear image
    %%    I_annotations = annotation data for this image
    %%
    %% Output:
    %%    features      = vector of ear features
	%%	
	%% If you are doing parameter optimization, use:
	%% global awet;
    %% awet.current_parameter_set
    
    %% Sample code for feature extraction:
    
    % Define awet as global so you can use parameters.
    global awet;
    
    % In awet.current_parameter_set is the current set of parameters.
    structOfActiveParams = awet.current_parameter_set;
    
    % Replace aleph with the name of your parameter; you can have as many
    % different parameters as you wish.
    specificParam = structOfActiveParams.aleph;
    
    % Code for feature extraction comes in here - use parameters if needed 
    % and annotations if needed.
    
    % features vector (as the output of this function) needs to be of size
    % 1 x n. This line of code here only transforms the input image into
    % the vector - plain image feature extractor:
    features = uint8(reshape(I, 1, []));
 end