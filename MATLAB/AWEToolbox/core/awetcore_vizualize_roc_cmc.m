function awetcore_vizualize_roc_cmc(distances, mMin, mMax)
    global awet;
   
    VER01FAR_cutoff = 1000;
    addpath(genpath('libraries/PhD_tool'));
    addpath(genpath('libraries/vito'));
    
    len = size(distances, 2);
    
    metrics = struct();
    metrics.ROC_X = [];
    metrics.ROC_Y = [];
    metrics.ROC_T = [];
    metrics.ROC_AUC = zeros(len, 1);
    metrics.ROC_OPTROCPT = [];
    
    metrics.ver_rate = [];
    metrics.miss_rate = [];
    metrics.EER_er = zeros(len, 1);
    metrics.VER_01FAR_ver = zeros(len, 1);
    metrics.VER_1FAR_ver = zeros(len, 1);
    
    metrics.true_scores = cell(len, 1);
    metrics.false_scores = cell(len, 1);
    
    metrics.rec_rates = cell(len, 1);
    metrics.ranks = cell(len, 1);
    metrics.rank_one_recognition_rate = zeros(len, 1);
    
    if awet.use_phd
        awetcore_log('USING PHD ROC and CMC', 2);
    else
        awetcore_log('USING MANUAL ROC and CMC', 2);
%         if ~exist('vl_version', 'file')
%             awetcore_log('vl_feat not yet installed, installing ...', 2);
%             run('libraries/vlfeat-0.9.20/toolbox/vl_setup');
%         end
    end
    
    for i = 1:len
        %% For PhD ROC
        outputs = distances(i).output;
        targets = distances(i).target;
%         disp(size(targets));
%         disp(sum(logical(targets)));
%         disp(sum(~logical(targets)));
%         disp('---');
        metrics.true_scores{i} = outputs(logical(targets));
        metrics.false_scores{i} = outputs(~logical(targets));
        
        %% PhD CMC
        classes_x = distances(i).classes_x;
        classes_y = distances(i).classes_y;
        outputs2D = distances(i).output2D;
        targets2D = distances(i).target2D;
        features_len = distances(i).features_len;
        
        if awet.use_phd
            cmcTmp = struct();
            cmcTmp.mode = 'all';
            if (awet.norm_mode == 1)
                cmcTmp.match_dist = outputs2D;
            elseif (awet.norm_mode == 2)
                cmcTmp.match_dist = outputs2D;
            end
            cmcTmp.same_cli_id = targets2D;
            cmcTmp.horizontal_ids = classes_x';
            cmcTmp.vertical_ids = classes_y';
            cmcTmp.dist = 'euc';
            cmcTmp.dim = features_len;

            [rec_rates, ranks] = produce_CMC_PhD(cmcTmp);

            metrics.rec_rates{i} = rec_rates;
            metrics.ranks{i} = ranks;
            metrics.rank_one_recognition_rate(i) = rec_rates(1);
        else        
            % DO NOT USE RIGHT NOW
            % Set diagonal so something low:
            if (awet.compressed_evaluation == 0)
                outputs2D(logical(eye(size(outputs2D)))) = -2;
            end

            lim = min(20, size(outputs2D, 2));
            ranks = 1:lim;
            rec_rates = zeros(lim, 1);
            allCount = size(outputs2D, 1);
            [~, sortedIndices] = sort(outputs2D, 2, 'descend');
            for r = 1:lim
                corr = 0;
                for a = 1:allCount
                    lineIndices = sortedIndices(a, :);
                    lineTargets = targets2D(a, lineIndices(1:r));
                    m = sum(lineTargets);
                    if (m >= 1)
                        corr = corr + 1;
                    end
                end
                rec_rates(r) = corr / allCount;
            end
            %metrics.rec_rates = [metrics.rec_rates rec_rates];
            %metrics.ranks = [metrics.ranks; ranks];
            metrics.rec_rates{i} = rec_rates';
            metrics.ranks{i} = ranks';
            metrics.rank_one_recognition_rate(i) = rec_rates(1);
        end
        
    end
    
    
    %% ROC
    [ver_rateF, miss_rate, rates_and_threshs] = produce_ROC_PhD_kfolds(metrics.true_scores, metrics.false_scores, 5000,0);
    ver_rate = ver_rateF.mean;
    ver_rate0 = ver_rateF.min;
    ver_rate1 = ver_rateF.max;

    ROC_data = struct();
    ROC_data.x = miss_rate;
    ROC_data.y = ver_rate;
    ROC_data.y0 = ver_rate0;
    ROC_data.y1 = ver_rate1;
    
    %awetcore_add_plot_and_draw_all('ROC', miss_rate, ver_rate, ver_rate0, ver_rate1, 'False Acceptance Rate', 'True Acceptance Rate', [awet.current_extractor.name, awet.parameter_path]);
    %awetcore_add_plot_and_draw_all('ROClog', miss_rate, ver_rate, ver_rate0, ver_rate1, 'False Acceptance Rate', 'True Acceptance Rate', [awet.current_extractor.name, awet.parameter_path]);
    %awetcore_add_plot_and_draw_all('ROC_crop', miss_rate, ver_rate, ver_rate0, ver_rate1, 'False Acceptance Rate', 'True Acceptance Rate', [awet.current_extractor.name, awet.parameter_path]);
    %awetcore_add_plot_and_draw_all('ROClog_crop', miss_rate, ver_rate, ver_rate0, ver_rate1, 'False Acceptance Rate', 'True Acceptance Rate', [awet.current_extractor.name, awet.parameter_path]);

    awetcore_add_plot_and_draw_all('ROClog', ROC_data.x, ROC_data.y, ROC_data.y0, ROC_data.y1, 'False Acceptance Rate', 'True Acceptance Rate', [awet.current_extractor.name, awet.parameter_path]);
    awetcore_add_plot_and_draw_all('ROC', ROC_data.x, ROC_data.y, ROC_data.y0, ROC_data.y1, 'False Acceptance Rate', 'True Acceptance Rate', [awet.current_extractor.name, awet.parameter_path]);
   
    %% CMC
    lim = min(cellfun('size', metrics.rec_rates, 2));
    rates_len = size(metrics.rec_rates, 1);
    rec_rates_all = zeros(rates_len, lim);
    ranks = metrics.ranks{1}(1:lim)';
    for m = 1:rates_len
        tmp = metrics.rec_rates{m};
        rec_rates_all(m, :) = tmp(1:lim);
    end
    rec_rates = mean(rec_rates_all, 1);
    rec_rates_std = std(rec_rates_all, 1, 1);
    rec_rates0 = rec_rates - rec_rates_std;
    rec_rates1 = rec_rates + rec_rates_std;
    
    CMC_data = struct();
    CMC_data.x = ranks;
    CMC_data.y = rec_rates;
    CMC_data.y0 = rec_rates0;
    CMC_data.y1 = rec_rates1;
    
    %awetcore_add_plot_and_draw_all('CMC', ranks, rec_rates, rec_rates0, rec_rates1, 'Rank', 'Recognition Rate', [awet.current_extractor.name, awet.parameter_path]);
    %awetcore_add_plot_and_draw_all('CMC_crop', ranks, rec_rates, rec_rates0, rec_rates1, 'Rank', 'Recognition Rate', [awet.current_extractor.name, awet.parameter_path]);
    awetcore_add_plot_and_draw_all('CMC', CMC_data.x, CMC_data.y, CMC_data.y0, CMC_data.y1, 'Rank', 'Recognition Rate', [awet.current_extractor.name, awet.parameter_path]);

    %% Results to struct
    data = struct();
    data.AUC =                  100 * rates_and_threshs.AUC;
    data.AUC_std =              100 * rates_and_threshs.AUC_std;
    data.AUC_num =                    rates_and_threshs.AUC_num;
    data.EER_er =               100 * rates_and_threshs.EER_er;
    data.EER_er_std =           100 * rates_and_threshs.EER_er_std;
    data.EER_num =                 rates_and_threshs.EER_num;
    data.VER_01FAR_ver =        100 * rates_and_threshs.VER_01FAR_ver;
    data.VER_01FAR_ver_std =    100 * rates_and_threshs.VER_01FAR_ver_std;
    data.VER_01FAR_num =          rates_and_threshs.VER_01FAR_num;
    data.VER_1FAR_ver =         100 * rates_and_threshs.VER_1FAR_ver;
    data.VER_1FAR_ver_std =     100 * rates_and_threshs.VER_1FAR_ver_std;
    data.VER_1FAR_num =           rates_and_threshs.VER_1FAR_num;
    data.rank_one_recognition_rate =     100 * mean(metrics.rank_one_recognition_rate, 1);
    data.rank_one_recognition_rate_std = 100 * std(metrics.rank_one_recognition_rate, 1);
    data.rank_one_recognition_rate_num = size(rec_rates_all, 1);
    
    num = 0;
    for m=1:size(metrics.true_scores,1)
        num = num + length(metrics.true_scores{m});
        num = num + length(metrics.false_scores{m});
    end
    data.num = num;
    
    epsilon = 10^-10;
    if (data.AUC_std < epsilon || data.EER_er_std < epsilon || data.VER_1FAR_ver_std < epsilon || data.rank_one_recognition_rate_std < epsilon )
         awetcore_log('ERROR! STD = 0', 0);
    end
    
    %% Format the results
    datap = struct();
    datap.AUC = [sprintf('%4.2f',data.AUC),'+-',sprintf('%4.2f',data.AUC_std), ' (', num2str(data.AUC_num), ')'];
    datap.EER = [sprintf('%4.2f',data.EER_er),'+-',sprintf('%4.2f',data.EER_er_std), ' (', num2str(data.EER_num), ')'];
    datap.VER_01FAR = [sprintf('%4.2f',data.VER_01FAR_ver),'+-',sprintf('%4.2f',data.VER_01FAR_ver_std), ' (', num2str(data.VER_01FAR_num), ')'];
    datap.VER_1FAR = [sprintf('%4.2f',data.VER_1FAR_ver),'+-',sprintf('%4.2f',data.VER_1FAR_ver_std), ' (', num2str(data.VER_1FAR_num), ')'];
    datap.rank_one = [sprintf('%4.2f',data.rank_one_recognition_rate),'+-',sprintf('%4.2f',data.rank_one_recognition_rate_std), ' (', num2str(data.rank_one_recognition_rate_num), ')'];
    datap.mfsc = num2str(rates_and_threshs.min_false_score_samples);
    datap.mtsc = num2str(rates_and_threshs.min_true_score_samples);
    datap.msc = num2str(rates_and_threshs.min_score_samples);
    datap.sc = num2str(data.num);
    
    if (rates_and_threshs.min_score_samples < VER01FAR_cutoff)
        datap.VER_01FAR = '/';
    end
    
    %% Results to JSON
    results_to_json = false;
    if (results_to_json)
        results_folder = strcat(awet.current_database.path, '-', ...
            awet.current_extractor.path, '-', awet.current_model.path, ...
            '-', awet.current_model.mode(1:3), '/');
        results_file = strcat(awet.run_id, '.json');
        awetcore_write_json([awet.results_path, results_folder], results_file, data);
        awetcore_log(['Results stored into ''', awet.results_path, results_folder, ''''], 0);
    end
    
    %% Results to Console
    awetcore_log(['\n********RESULTS********', ...
        '\nRank-1\t\t\t: ', datap.rank_one, ...
        '\nEER_er\t\t\t: ', datap.EER, ...
        '\nVER_01FAR_ver\t\t: ', datap.VER_01FAR, ...
        '\nVER_1FAR_ver\t\t: ', datap.VER_1FAR, ...
        '\nAUC\t\t\t: ', datap.AUC, ...
        '\n#min(FSc/n)\t\t: ', datap.mfsc, ...
        '\n#min(TSc/n)\t\t: ', datap.mtsc, ...
        '\n#min(Sc/n)\t\t: ', datap.msc, ...
        '\n#Sc\t\t: ', datap.sc, ...
        '\n***********************'
    ], 0);
    
    %% Results to File
    results_to_file = true;
    if (results_to_file)
        if (awet.results_header_set == 0)
            awet.results_header_set = 1;
            awetcore_log_file('P\tDB\tM\tParam\tRank-1\tEER\tVER01FAR\tVER1FAR\tAUC\t#min(FSc/n)\t#min(TSc/n)\t#min(Sc/n)\t#Sc');
        end
        awetcore_log_file(['P', num2str(awet.current_database.protocol), '\t', awet.current_database.name, '\t', awet.current_extractor.name, '\t', awet.parameter_path, ...
            '\t', datap.rank_one, ...
            '\t', datap.EER, ...
            '\t', datap.VER_01FAR, ...
            '\t', datap.VER_1FAR, ...
            '\t', datap.AUC, ...
            '\t: ', datap.mfsc, ...
            '\t: ', datap.mtsc, ...
            '\t: ', datap.msc, ...
            '\t', datap.sc ...
        ]);
    end
    
    %% Results to RAW file
    %store_raw_data(data, filename_plain, x, y, y0, y1);
    store_raw_data(data, ROC_data, CMC_data);
    
    %% Pack the results into ZIP
    dirn = [awet.results_path, awet.run_id, '-plots/'];
    zipn = [awet.results_path, awet.run_id, '-plots.zip'];
    zip(zipn, {'*'}, dirn);
end

function store_raw_data(data, ROC_data, CMC_data)
    % store the following to plain text 
    % x, y, y0, y1 
    % accuracy measures
    
    % check and create folder
    fname = awetcore_check_dir_raw('raw');
    
    fid = fopen(fname, 'wt');
    fprintfs(fid, ROC_data.x);
    fprintfs(fid, ROC_data.y);
    fprintfs(fid, ROC_data.y0);
    fprintfs(fid, ROC_data.y1);
    
    fprintfs(fid, CMC_data.x);
    fprintfs(fid, CMC_data.y);
    fprintfs(fid, CMC_data.y0);
    fprintfs(fid, CMC_data.y1);
    
    datakeys = fieldnames(data);
    for i = 1:numel(datakeys)
        key = datakeys{i};
        fprintfs(fid, [key, '=', num2str(data.(key))], 1);
    end
    fclose(fid);
end

function fprintfs(fid, s, sent)
    if ~exist('sent','var') || isempty(sent)
        fprintf(fid,'%d,',s);
    else
        fprintf(fid,'%s',s);
    end
    fprintf(fid,'%s\n','');
end