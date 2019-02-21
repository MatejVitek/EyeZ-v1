%% K-fold example
% generiraj 5 foldov scoreov za kliente
k = 5;
n_client_scores = 50;
true_scores = zeros(k,n_client_scores);
for i=1:k
    true_scores(i,:) = 0.8-i*0.1 + 0.8.*randn(n_client_scores,1);
end

% generiraj 5 foldov impostor scorov
n_imp_scores = 250;
false_scores = zeros(k,n_imp_scores);
for i=1:k
    false_scores(i,:) = 3+i*0.1 + .7.*randn(n_imp_scores,1);
end

%save('outputs.mat', 'true_scores', 'false_scores');
load('outputs.mat');

true_scoresx = cell(k,1);
false_scoresx = cell(k,1);

for m = 1:k
    true_scoresx{m} = true_scores(m,:);
    false_scoresx{m} = false_scores(m,:);
end

false_scoresx{1} = [false_scoresx{1} 0.7111];

[ver_rate, miss_rate, rates_and_threshs] = produce_ROC_PhD_kfolds(true_scoresx, false_scoresx, 5000,1);
figure
disp(sprintf('EER is: %f+-%f',rates_and_threshs.EER_er, rates_and_threshs.EER_er_std));
disp(sprintf('VER at 0.1 percent FAR is: %f+-%f',rates_and_threshs.VER_01FAR_ver, rates_and_threshs.VER_01FAR_ver_std));
disp(sprintf('VER at 1 percent FAR is: %f+-%f',rates_and_threshs.VER_1FAR_ver, rates_and_threshs.VER_1FAR_ver_std));
set(gca, 'XScale', 'log')
h=plot_ROC_PhD_kfolds(ver_rate, miss_rate, 'r', 2);


%% Bootstraping example

% generiraj 1 foldov scoreov za kliente
k = 1;
n_client_scores = 1000;
true_scores = zeros(k,n_client_scores);
for i=1:k
    true_scores(i,:) = 0.8-i*0.1 + 0.8.*randn(n_client_scores,1);
end

% generiraj 1 foldov impostor scorov
n_imp_scores = 50000;
false_scores = zeros(k,n_imp_scores);
for i=1:k
    false_scores(i,:) = 3+i*0.1 + .7.*randn(n_imp_scores,1);
end

[ver_rate, miss_rate, rates_and_threshs] = produce_ROC_PhD_boot(true_scores, false_scores, 5000,60,20,0); %vzamem 60% podatkov in generiram 20 bootstrap setov

disp(sprintf('EER is: %f+-%f',rates_and_threshs.EER_er, rates_and_threshs.EER_er_std));
disp(sprintf('VER at 0.1 percent FAR is: %f+-%f',rates_and_threshs.VER_01FAR_ver, rates_and_threshs.VER_01FAR_ver_std));

h=plot_ROC_PhD_kfolds(ver_rate, miss_rate, 'r', 2);
