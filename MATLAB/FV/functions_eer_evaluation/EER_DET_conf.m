function [EER, confInterEER, OP, confInterOP, plots, FMR1000, ZeroFMR, FNMR1000, ZeroFNMR] = ...
    EER_DET_conf(clients,imposteurs,OPvalue,pas0,varargin)

% function: EER_DET_conf
%
% DESCRIPTION:
% It plots traditional curves and gives also some interesting values in 
% order to evaluate the performance of a biometric verification system. 
% The curves are:
%       - Receiver Operating Characteristic (ROC) curve
%       - Detection Error Trade-off (DET) curve
%       - FAR vs FRR
% The values are:
%       - Equal Error Rate (EER) which is computed as the point where
%       FAR=FRR
%       - Operating Point (OP) which is defined in terms of FRR (%)
%       achieved for a fixed FAR
% A 90% interval of confidence is provided for both values (parametric 
% method).
%
% INPUTS:
% clients: vector of genuine/client scores
% imposteurs: vector of impostor scores
% OPvalue: value of FAR at which the OP value is estimated
% pas0: number of thresholds used the estimate the score distributions
% (10000 is advised for this parameter)
%
% OUTPUTS:
% EER: EER value
% confInterEER: error margin on EER value
% OP: OP value
% confInterOP: error margin on OP value
%
%
% CONTACT: aurelien.mayoue@int-edu.eu
% 19/11/2007

% Addition:
%% Use Input Parser for optional arguments
    p = inputParser;
    p.KeepUnmatched = true;
    addRequired(p, 'clients');
    addRequired(p, 'imposteurs');
    addRequired(p, 'OPvalue');
    addRequired(p, 'pas0');
    addParamValue(p, 'ShowPlots', 'Yes', @ischar);
    addParamValue(p, 'DetermineConfInter', 'Yes', @ischar);
    addParamValue(p, 'DetermineOP', 'Yes', @ischar);
    addParamValue(p, 'CalculateFMR', 'No', @ischar);
    addParamValue(p, 'CalculateFNMR', 'No', @ischar);
    addParamValue(p, 'k', '1', @ischar);
    
    parse(p, clients, imposteurs, OPvalue, pas0, varargin{:});
    
    determineOP = strcmpi('Yes', p.Results.DetermineOP);
    determineConfInter = strcmpi('Yes', p.Results.DetermineConfInter);
    calculateFMR = strcmpi('Yes', p.Results.CalculateFMR);
    calculateFNMR = strcmpi('Yes', p.Results.CalculateFNMR);


%%%%% estimation of thresholds used to calculate FAR et FRR

% maximum of client scores
m0 = max (clients);

% size of client vector
num_clients = length (clients);

% minimum impostor scores
m1 = min (imposteurs);

% size of impostor vector
num_imposteurs = length (imposteurs);

% calculation of the step
pas1 = (m0 - m1)/pas0;
x = [m1:pas1:m0]';

num = length (x);

%%%%%

%%%%% calculation of FAR and FRR

for i=1:num
    fr=0;
    fa=0;
    for j=1:num_clients
        if clients(j)<x(i)
            fr=fr+1;
        end
    end
    for k=1:num_imposteurs
        if imposteurs(k)>=x(i)
            fa=fa+1;
        end
    end
    FRR(i)=100*fr/num_clients;
    FAR(i)=100*fa/num_imposteurs;
end 

%%%%%

%%%%% calculation of EER value

tmp1=find (FRR-FAR<=0);
tmps=length(tmp1);

if ((FAR(tmps)-FRR(tmps))<=(FRR(tmps+1)-FAR(tmps+1)))
    EER=(FAR(tmps)+FRR(tmps))/2;tmpEER=tmps;
else
    EER=(FRR(tmps+1)+FAR(tmps+1))/2;tmpEER=tmps+1;
end

%%%%%

if determineConfInter
    %%%%% calculation of the confidence intervals
    [FARconfMIN  FRRconfMIN FARconfMAX FRRconfMAX]=ParamConfInter(FAR/100,FRR/100,num_imposteurs,num_clients);

    % EER
    confInterEER=EER-100*(FARconfMIN(tmpEER)+FRRconfMIN(tmpEER))/2;
end

% If OP should be determined or not
if determineOP
    %%%%% calculation of the OP value
    tmp2=find (OPvalue-FAR<=0);
    tmpOP=length(tmp2);

    OP=FRR(tmpOP);
    %%%%%
    
    % Operating Point Confidence Interval
    confInterOP=OP-100*FRRconfMIN(tmpOP);

    %%%%
else
    tmpOP = [];  % Just for plotting
    OP = [];
    confInterOP = [];
end

% Determine FMR1000 and ZeroFMR
if calculateFMR
    % Calculate FMR1000
    tmp2=find(FAR>0.1);
    tmpFMR1000=length(tmp2);
    FMR1000=FRR(tmpFMR1000);
    % Calculate ZeroFMR
    tmp2=find(FAR>0);
    tmpZeroFMR=length(tmp2);
    ZeroFMR=FRR(tmpZeroFMR);
else
    tmpFMR1000 = [];
    FMR1000 = [];
    tmpZeroFMR = [];
    ZeroFMR = [];
end

% Determine FNMR1000 and ZeroFNMR
if calculateFNMR
    % Calculate FNMR1000
    tmp2=find(FRR<=0.1);
    tmpFNMR1000=length(tmp2);
    FNMR1000=FAR(tmpFNMR1000);
    % Calculate ZeroFNMR
    tmp2=find(FRR==0);
    tmpZeroFNMR=length(tmp2);
    ZeroFNMR=FAR(tmpZeroFNMR);
else
    tmpFNMR1000 = [];
    FNMR1000 = [];
    tmpZeroFNMR = [];
    ZeroFNMR = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% If Plots should not be shown, just return values
if (strcmpi('No', p.Results.ShowPlots))
    plots = struct('x', x, 'FRR', FRR, 'FAR', FAR, 'tmps', tmps, 'tmpOP', tmpOP);
    if calculateFMR
        plots.tmpFMR1000 = tmpFMR1000;
        plots.FMR1000 = FMR1000;
        plots.tmpZeroFMR = tmpZeroFMR;
        plots.ZeroFMR = ZeroFMR;
    end
    if calculateFNMR

    end
    return;
end

%%%%% plotting of curves

% FAR vs FRR
figure(1);
plot (x,FRR,'r');
hold on;plot (x,FAR,'b');
% Plot FMR1000 and ZeroFMR
if ~isempty(tmpFMR1000) && ~isempty(tmpZeroFMR)
    hold on;scatter (x(tmpFMR1000),FRR(tmpFMR1000),'og');
    hold on;scatter (x(tmpZeroFMR),FRR(tmpZeroFMR),'oy');
end
% Plot FNMR1000 and ZeroFNMR
if ~isempty(tmpFNMR1000) && ~isempty(tmpZeroFNMR)
    hold on;scatter (x(tmpFNMR1000),FAR(tmpFNMR1000),'*g');
    hold on;scatter (x(tmpZeroFNMR),FAR(tmpZeroFNMR),'*y');
end
xlabel ('Threshold');
ylabel ('Error');
title ('FAR vs FRR graph');

% interpolation for the plotting
equaX=x(tmps)*(FRR(tmps+1)-FAR(tmps+1))+x(tmps+1)*(FAR(tmps)-FRR(tmps));
equaY=FRR(tmps+1)-FAR(tmps+1)+FAR(tmps)-FRR(tmps);
threshold=equaX/equaY;
EERplot=threshold*(FAR(tmps)-FAR(tmps+1))/(x(tmps)-x(tmps+1))+(x(tmps)*FAR(tmps+1)-x(tmps+1)*FAR(tmps))/(x(tmps)-x(tmps+1));

% ROC curve
figure(2);
plot (FAR,100-FRR,'r');
xlabel ('Impostor Attempts Accepted = FAR (%)');
ylabel ('Genuine Attempts Accepted = 1-FRR (%)');
title ('ROC curve');
hold on;scatter (EERplot,100-EERplot,'ok');
% Plot OP
if ~isempty(tmpOP)
    hold on;scatter (FAR(tmpOP),100-FRR(tmpOP),'xk');
end
axis ([0 50 50 100]);

% DET curve
figure(3);
h = Plot_DET(FRR/100,FAR/100,'r');
hold on; Plot_DET(EERplot/100,EERplot/100,'ok');
% Plot OP
if ~isempty(tmpOP)
    hold on; Plot_DET(FRR(tmpOP)/100,FAR(tmpOP)/100,'xk');
end
title ('DET curve');

% Return the plotting data
plots = struct('x', x, 'FRR', FRR, 'FAR', FAR, 'tmps', tmps, 'tmpOP', tmpOP);
if calculateFMR
    plots.tmpFMR1000 = tmpFMR1000;
    plots.FMR1000 = FMR1000;
    plots.tmpZeroFMR = tmpZeroFMR;
    plots.ZeroFMR = ZeroFMR;
end
if calculateFNMR
    plots.tmpFNMR1000 = tmpFNMR1000;
    plots.FNMR1000 = FNMR1000;
    plots.tmpZeroFNMR = tmpZeroFNMR;
    plots.ZeroFNMR = ZeroFNMR;
end
