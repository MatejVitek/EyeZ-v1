% *************************************************************************
% This file is part of the feature level fusion framework for finger vein
% recognition (MATLAB implementation). 
%
% Reference:
% Advanced Variants of Feature Level Fusion for Finger Vein Recognition 
% C. Kauba, E. Piciucco, E. Maiorana, P. Campisi and A. Uhl 
% In Proceedings of the International Conference of the Biometrics Special 
% Interest Group (BIOSIG'16), pp. 1-12, Darmstadt, Germany, Sept. 21 - 23
%
% Authors: Christof Kauba <ckauba@cosy.sbg.ac.at> and 
%          Emanuela Piciucco <emanuela.piciucco@stud.uniroma3.it>
% Date:    31th August 2016
% License: Simplified BSD License
%
% 
% Description:
% This function computes the genuine and impostor scores based on matching
% all the images inside the UTFVP dataset.
%
% Parameters:
%  features - A cell array (60x6x4) containing the feature images
%  varargin - Additional (optional) input parameters:
%      MatchMode - either full, train or test ... determines nr. matches
%
% Returns:
%  genuine_scores - A vector containing all genuine scores
%  impostor_scores - A vector containing all impostor scores
% *************************************************************************

function [ genuine_scores, impostor_scores ] = computeScoresFull(features, varargin)
%COMPUTESCORESFULL Computes matching scores using Miura matcher
%   Detailed explanation goes here
    global use_progress_bar;
	% Get the settings
	global settings;
	ls = settings.MatchingSettings;	% local settings
    
	matchMode_default = 'full';
    
    img_per_pers = ls.no_fing*ls.img_per_finger;
	total_imgs = ls.no_users*img_per_pers;
	
	%% Get additional options using inputparser
	p = inputParser;
	p.KeepUnmatched = true;
	addParamValue(p, 'MatchMode', matchMode_default, @ischar);
	addParamValue(p, 'cw', ls.cw_default, @isscalar);
	addParamValue(p, 'ch', ls.ch_default, @isscalar);

	parse(p, varargin{:});
	params = p.Results;

	matchMode = params.MatchMode;
	cw = params.cw;
	ch = params.ch;
	
	% Reshape the feature cell array for convenient operation
    veins = reshape_features(features, img_per_pers, ls.no_fing, ls.img_per_finger, ls.no_users);
	
	% Prepare matching set (full/train/test)
	train_set_i = zeros(35*4, 1); % Indices of data for training set
	k = 0; % Finger number counter
	for i=1:img_per_pers:35*img_per_pers
		train_set_i(k*4+1:k*4+4)=[i + mod(k,6)*4:i + mod(k,6)*4 + 3];
		%i + mod(k,6)*4
		k=k+1;    
	end
	real_set_i = setdiff([1:total_imgs], train_set_i);
	
	% Select set for operation
	switch matchMode
		case 'full'
			% veins = veins;
			fprintf('Match mode: full set\n');
		case 'train'
			veins = veins(train_set_i);
			fprintf('Match mode: training set\n');
		case 'test'
			veins = veins(real_set_i);
			fprintf('Match mode: test set\n');
	end

    % Check if the features are probabilistic (Staple(r)/Collate) or not
    if length(size(veins{1})) == 4
        fprintf('Preparing prob feature files...\n');
        veins = cellfun(@(x) x(:,:,1,2), veins, 'UniformOutput', false);
    end

    % Convert to single if logical
    if isa(veins{1}, 'logical');
        veins = cellfun(@(x) single(x), veins, 'UniformOutput', false);
    end
    
    %sim martix calculation
    if use_progress_bar
        ProgressBar.update('new', 'Computing Scores ...', 'Score calculation');
        starttime = tic;
    end
    fprintf('Calculating similarity matrix\n');
    nrFiles = length(veins);
    nrCompares = nrFiles^2;
    sim_matrix = zeros(nrFiles, nrFiles);

    for i=1:length(veins)
        for j=1:i % Half
            if i==j
                sim_matrix(i,j) = 1;
            else
                sim_matrix(i,j) =  miura_match(veins{i},veins{j}, cw, ch);
            end
            if use_progress_bar
                updateStatus((i-2)*(i-1) + j, nrCompares, toc(starttime));
            end
        end
    end
    
    if use_progress_bar
        ProgressBar.update('close');
    end
    
    % Computing the actual scores
    fprintf('Computing scores...\n');
    [ genuine_scores, impostor_scores ] = compute_score2( sim_matrix );
end

% Helper functions
function [ gen_score, imp_score ] = compute_score2( sim_matrix )
%compute_score2 Computes the genuine and impostor scores from a given similarity matrix
%   Detailed explanation goes here
    gen_score = [];
    imp_score = [];
    for i=1:size(sim_matrix,1)
        fingi = floor((i-1)/4);
        for j=1:i % Half
            fingj = floor((j-1)/4);
            if( (fingi==fingj) )
                if (i ~= j)
                    gen_score(end+1) = sim_matrix(i,j);
                end
            else
                imp_score(end+1) = sim_matrix(i,j);
            end
        end
    end
end

function [ veins_res ] = reshape_features( features,img_per_pers, no_fing,img_per_finger, no_users)
%reshape_features Reshapes the vein feature cell array such that it is more convenient to handle
%   Detailed explanation goes here
    veins1=cell(img_per_pers,no_users);

    for i=1:no_users
       v=cell(no_fing,img_per_finger);
       v(:,:)=features(i,:,:);
       v=v';
       v=reshape(v,[],1);
       veins1(:,i)=v;
       clear v
    end

    veins_res= reshape(veins1,[],1);
end
