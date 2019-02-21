% The function produces ROC curve data from genuine and impostor scores
% 
% PROTOTYPE
% [ver_rate, miss_rate, rates_and_threshs] = produce_ROC_PhD_boot(true_scores, false_scores, resolu,percent,n_boots,flag)
% 
% flag = 1 poemni da raèunamo in rišemo povpreèje in 95% intervale zapuanja
% flag = 0 pomeni da rišemo mean +- std
%
% Macskassy, S., Provost, F., 2004. Confidence bands for ROC curves:
% Methods and an empirical study. In: Proc. First Workshop on ROC
% Analysis in AI (ROCAI-04).
%
%  
% 
% INPUTS:
% true_scores           - a vector of genuine/client/true scores, which is 
%                         expected to have been produced using a distance 
%                         measure and not a similarity measure (obligatory 
%                         argument)
% false_scores          - a vector of impostor/false scores, which is 
%                         expected to have been produced using a distance 
%                         measure and not a similarity measure (obligatory 
%                         argument) 
% resolution            - a parameter determining the number of points at
%                         which to compute the ROC curve data; default=2500 
%                         (optional argument)
%
%
% Copyright (c) 2016 Vitomir Štruc
% Faculty of Electrical Engineering,
% University of Ljubljana, Slovenia
% http://luks.fe.uni-lj.si/en/staff/vitomir/index.html
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files, to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.
% 
% Januar 2016

function [ver_rate, miss_rate, rates_and_threshs] = produce_ROC_PhD_boot(true_scores, false_scores, resolu,percent,n_boots,flag)

[~,n_true] = size(true_scores);
[~,n_false] = size(false_scores);

num_true = round(percent/100*n_true);
num_false = round(percent/100*n_false);


trues = zeros(n_boots,num_true);
falses = zeros(n_boots,num_false);
for i=1:n_boots
    indt = randperm(n_true);
    indf = randperm(n_false);
    
    trues(i,:) = true_scores(indt(1:num_true));
    falses(i,:) = false_scores(indf(1:num_false));    
end

[ver_rate, miss_rate, rates_and_threshs] = produce_ROC_PhD_kfolds(trues, falses, resolu,flag);



























