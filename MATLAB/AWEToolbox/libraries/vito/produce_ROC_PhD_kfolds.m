function [ver_rate, miss_rate, rates_and_threshs] = produce_ROC_PhD_kfolds(true_scores, false_scores, resolu,flag)

MODE = 1;

%% Init 
ver_rate = [];
miss_rate = [];
rates_and_threshs = [];
n_folds = size(false_scores,1);

%% Init operations
if MODE == 0
    cli_scor = size(true_scores,2);
    imp_scor = size(false_scores,2);
    dmax = true_scores;
    dmin = false_scores;    
end

%% Compute ROC curve data
min_false_score_samples = realmax;
min_true_score_samples = realmax;
min_score_samples = realmax;
% get maximum and minimum value - this ensures threshold alignment
if MODE == 0 
    dminx = min(min(false_scores))-0.1;
    dmaxx = max(max(true_scores))+0.1;
else
    dminx = realmax;
    dmaxx = realmin;
    for m=1:n_folds
        mi = min(false_scores{m});
        mx = max(true_scores{m});

        if mi < dminx
            dminx = mi;
        end
        if mx > dmaxx
            dmaxx = mx;
        end
        
        sizF = size(false_scores{m}, 1);
        sizT = size(true_scores{m}, 1);
        siz = sizF + sizT;
        if (sizF < min_false_score_samples)
            min_false_score_samples = sizF;
        end
        if (sizT < min_true_score_samples)
            min_true_score_samples = sizT;
        end
        if (siz < min_score_samples)
            min_score_samples = siz;
        end
    end
    dminx = dminx - 0.1;
    dmaxx = dmaxx + 0.1;
end

z_cr = 1.96;

%computing FRE curve
delta = (dmaxx-dminx)/resolu;
counter=1;
fre = zeros(n_folds,resolu);
fae = zeros(n_folds,resolu);
for trash=dminx:delta:dmaxx
    if MODE == 0 
        num_ok = sum(true_scores<trash,2); % za vse folde za vse elemente
        fre(:,counter) = 1-(num_ok/cli_scor);
        
        num_ok = sum(dmin<trash,2); % za vse folde za vse elemente
        fae(:,counter) = (num_ok/imp_scor);
    else
        for m=1:n_folds
            num_ok = sum(true_scores{m} < trash);
            cli_scor = length(true_scores{m});
            fre(m,counter) = 1 - (num_ok/cli_scor);
        end    
        for m=1:n_folds
            num_ok = sum(false_scores{m} < trash);
            imp_scor = length(false_scores{m});
            fae(m,counter) = (num_ok/imp_scor);
        end
    end   
    counter = counter+1;
end

% compute mean and std
fre_m = mean(fre);
fae_m = mean(fae);
fre_std = std(fre,1);
fae_std = std(fae,1);

if flag ==1
    confidence_95_fre = z_cr*fre_std/sqrt(n_folds);
    confidence_95_fae = z_cr*fae_std/sqrt(n_folds);
else
    confidence_95_fre = fre_std;
    confidence_95_fae = fae_std;
end
fre_min = fre_m-confidence_95_fre;
fre_max = fre_m+confidence_95_fre;
fae_min = fae_m-confidence_95_fae;
fae_max = fae_m+confidence_95_fae;

%% Computing characteristic error rates and corresponding thresholds

% % %Minimal HTER
C=fae_m+fre_m;
% % [dummy,index] = min(C);
% % rates_and_threshs.minHTER_er  = C(index)/2;
% % rates_and_threshs.minHTER_tr  = dminx+(index-1)*delta;
% % rates_and_threshs.minHTER_frr = sum(dmax>(dminx+(index-1)*delta))/cli_scor;
% % rates_and_threshs.minHTER_ver = 1-rates_and_threshs.minHTER_frr;
% % rates_and_threshs.minHTER_far = sum(dmin<(dminx+(index-1)*delta))/imp_scor;


%EER, FRR = 0.1FAR, FRR = 10FAR, @0.01%FAR, @0.1%FAR, @1%FAR
maxiEER = Inf;
maxi01p = Inf;
maxi1p = Inf;

maxiEERf = Inf*ones(1,n_folds);
indexiEER = zeros(1,n_folds);

for i=1:resolu+1
    %% EER - Mean
    if abs(fae_m(i)-fre_m(i)) < maxiEER
       index1 = i;
       maxiEER = abs(fae_m(i)-fre_m(i));
    end
%     for j=1:n_folds
%         if abs(fae(j,i)-fre(j,i)) < maxiEERf(j);
%             indexiEER(j)=i;
%             maxiEERf(j) = abs(fae(j,i)-fre(j,i));
%         end
%     end
    
    %% VER @0.1% FAR - Mean
    if abs(fae_m(i)-0.1/100) < maxi01p
       index5 = i;
       maxi01p = abs(fae_m(i)-0.1/100);
    end
    
    %% VER @1% FAR - Mean
    if abs(fae_m(i)-1/100) < maxi1p
       index6 = i;
       maxi1p = abs(fae_m(i)-1/100);
    end
end

%% EER ER_STD
% eer = zeros(1, n_folds);
% Cx = fae + fre;
% for i=1:n_folds
%    eer(i) = Cx(i,indexiEER(i))/2;
% end
% er_std = std(eer, 1);
% rates_and_threshs.EER_er_std  = er_std;

er_tmp = C(index1)/2;
Cmin = fae_min + fre_min;
er_min = Cmin(index1)/2;
er_std = abs(er_tmp - er_min);
rates_and_threshs.EER_er_std  = er_std;


%% EER
frr = fre_m(index1);
ver = 1-frr;
%ver_std = abs(ver - (1 - fre_max(index1)));
far = fae_m(index1);

rates_and_threshs.EER_er  = C(index1)/2;
rates_and_threshs.EER_tr  = dminx+(index1-1)*delta;
rates_and_threshs.EER_frr = frr;
rates_and_threshs.EER_ver = ver;
%rates_and_threshs.EER_ver_std = ver_std;
rates_and_threshs.EER_far = far;
rates_and_threshs.EER_num = index1;


%% VER @0.1% FAR
frr = fre_m(index5);
ver = 1-frr;
ver_std = abs(ver - (1 - fre_max(index5)));
far = fae_m(index5);

rates_and_threshs.VER_01FAR_er  = C(index5)/2;
rates_and_threshs.VER_01FAR_tr  = dminx+(index5-1)*delta;
rates_and_threshs.VER_01FAR_frr = frr;
rates_and_threshs.VER_01FAR_ver = ver;
rates_and_threshs.VER_01FAR_ver_std = ver_std;
rates_and_threshs.VER_01FAR_far = far;
rates_and_threshs.VER_01FAR_num = index5;

%% VER @1% FAR
frr = fre_m(index6);
ver = 1-frr;
ver_std = abs(ver - (1 - fre_max(index6)));
far = fae_m(index6);

rates_and_threshs.VER_1FAR_er  = C(index6)/2;
rates_and_threshs.VER_1FAR_tr  = dminx+(index6-1)*delta;
rates_and_threshs.VER_1FAR_frr = frr;
rates_and_threshs.VER_1FAR_ver = ver;
rates_and_threshs.VER_1FAR_ver_std = ver_std;
rates_and_threshs.VER_1FAR_far = far;
rates_and_threshs.VER_1FAR_num = index6;

%% Number of false scores
rates_and_threshs.min_false_score_samples = min_false_score_samples;
rates_and_threshs.min_true_score_samples = min_true_score_samples;
rates_and_threshs.min_score_samples = min_score_samples;

%% set ver_rate, miss_rates
ver_rate.mean = 1-fre_m;
ver_rate.min = 1-fre_max;
ver_rate.max = 1-fre_min;
miss_rate = fae_m;

% cut
VER01FAR_cutoff = 1000;

if (rates_and_threshs.min_score_samples < VER01FAR_cutoff)
    ll = 1e-2;
else
    ll = 1e-3;
end

% We cutoff, but still leave one to the left in, because otherwise we
% start drawing a bi after the line, which is not ok
tt = miss_rate > ll;
ftt = find(tt);
ftt = ftt(1) - 1;
if (ftt > 0)
    tt(ftt) = 1;
    % move to exact line
    miss_rate(ftt) = ll;
end

miss_rate = miss_rate(tt);
ver_rate.mean = ver_rate.mean(tt);
ver_rate.min = ver_rate.min(tt);
ver_rate.max= ver_rate.max(tt);

%% AUC
auc_mean = trapz(miss_rate, ver_rate.mean);
auc_max = trapz(miss_rate, ver_rate.max);
auc_diff = abs(auc_mean - auc_max);
rates_and_threshs.AUC = auc_mean;
rates_and_threshs.AUC_std = auc_diff;
rates_and_threshs.AUC_num = size(miss_rate, 2);























