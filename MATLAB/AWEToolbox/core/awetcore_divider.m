function divisions = awetcore_divider(y)
% Divisions - either random or from files are done here
    global awet;

    kfold = awet.crossvalind.factor;
    curr_db_dir = [awet.databases_path, awet.current_database.path, '/'];

    if awet.current_database.protocol == 1
        awetcore_log('PROTOCOL 1: Random Distrubution', 1);

%         divisions = cell(kfold, 1);
%         for i = 1:kfold
%             [~, i_test] = awetcore_divide_db_type2(y, kfold);
%             divisions{i} = i_test;
%         end

        % Firstly we check whether distributions are already present, only
        % if they are not, we generate new random distribution
        
        % !!!! WARN TMP !!!
        distrFN = [curr_db_dir, awet.distribution_files.distr];
        if (exist(distrFN, 'file') == 2)
           load(distrFN);
        else
            divisions = awetcore_divide_db_type_forced(y, kfold);
           save(distrFN, 'divisions');
        end

    elseif awet.current_database.protocol == 2 || awet.current_database.protocol == 3
        trainFN = [curr_db_dir, awet.distribution_files.train];
        testFN = [curr_db_dir, awet.distribution_files.test];
        if (~(exist(trainFN, 'file') == 2) || ~(exist(testFN, 'file') == 2))
            awetcore_log('FATAL ERROR NO DISTRIBUTION FILE', 0);
            return;
        end

        if awet.current_database.protocol == 2
            awetcore_log('PROTOCOL 2: Distribution from file, train set', 1);
            distr = uint32(dlmread(trainFN, ' '));
            % train part of the data used
           
        elseif awet.current_database.protocol == 3
            awetcore_log('PROTOCOL 3: Distribution from file, test set with bootstrapping', 1);
            distr = uint32(dlmread(testFN, ' '));
            % test part of the data used
        end
            
        divisions = cell(size(distr, 1), 1);
        for i = 1:size(distr, 1)
            line = distr(i, :);
            line = line(line~=0);
            divisions{i} = line';
        end
    end
end
