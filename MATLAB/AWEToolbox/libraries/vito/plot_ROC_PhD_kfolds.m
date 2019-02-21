% The function plots the so-called ROC curve based on the provided input data
% 
% PROTOTYPE
% h=plot_ROC_PhD_kfolds(ver_rate, miss_rate, color, thickness)
% 
% Goes together with the produce_ROC_PhD_kfolds function
% 
% 
% Copyright (c) 2011 Vitomir Štruc
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
% November 2011
function h=plot_ROC_PhD_kfolds(ver_rate, miss_rate, color, thickness)

%% Init operations
h = [];

%% Check inputs

%check number of inputs
if nargin <2
    disp('Wrong number of input parameters! The function requires at least two input arguments.')
    return;
elseif nargin==2
    color='r';
    thickness=2;
elseif nargin==3
    thickness=2;    
end



%% Plot axis
h=semilogx(miss_rate,ver_rate.mean,'Color',color,'Linewidth',thickness);
hold on
X=[miss_rate,fliplr(miss_rate)];
Y=[ver_rate.min,fliplr(ver_rate.max)];
fill(X,Y,'r');


%axis labels
xlabel('False Accept Rate')
ylabel('Verification Rate')

%grid lines 
grid on
axis([1e-3 1 0 1])

%other 
set(gca, ...
  'XMinorTick'  , 'on'      , ...
  'YGrid'       , 'on'      , ...
  'XGrid'       , 'on'      , ...
  'YTick'      , 0:0.1:1 );































