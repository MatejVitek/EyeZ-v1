function Plot_EER_ROC_DET_Curves(x, FRR, FAR, tmps, tmpOP)
%PLOT_EER_ROC_DET_CURVES Plots FAR/FRR, ROC and DET curves
%   Detailed explanation goes here

%%%%% plotting of curves

% FAR vs FRR
figure(1);
plot (x,FRR,'r');
hold on;plot (x,FAR,'b');
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
hold on;scatter (FAR(tmpOP),100-FRR(tmpOP),'xk');
axis ([0 50 50 100]);

% DET curve
figure(3);
h = Plot_DET(FRR/100,FAR/100,'r');
hold on; Plot_DET(EERplot/100,EERplot/100,'ok');
hold on; Plot_DET(FRR(tmpOP)/100,FAR(tmpOP)/100,'xk');
title ('DET curve');

end

