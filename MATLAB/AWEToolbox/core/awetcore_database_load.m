function [db, annotation_data] = awetcore_database_load(path)
    global awet;
    len = 0;
    root = [awet.databases_path, path, '/'];
    
    % first run for allocation
    listing = dir(root);
    for i = 1:numel(listing)
        if (listing(i).isdir && ~strcmp(listing(i).name, '.') && ~strcmp(listing(i).name, '..'))
            f = dir(strcat(root, listing(i).name, '/*'));
            f = f(~cellfun(@isempty,regexpi({f.name}, ['.*(', strjoin(awet.file_endings, '|'), ')'])));
            len = len + numel(f);
        end
    end

    files = cell(len, 1);
    filenames = cell(len, 1);
    classes = uint32(zeros(len, 1));
    annotations = table();
    inx = 1;
    for i = 1:numel(listing)
        if (listing(i).isdir && ~strcmp(listing(i).name, '.') && ~strcmp(listing(i).name, '..'))
            person_id = listing(i).name;
            inner_dir = strcat(root, person_id, '/');
            listing_f = dir(strcat(inner_dir, '*'));
            listing_f = listing_f(~cellfun(@isempty,regexpi({listing_f.name}, ['.*(', strjoin(awet.file_endings, '|'), ')'])));
            annotations_f = strcat(inner_dir, awet.annotations_file_name);
            
            for j = 1:numel(listing_f)
                fname = cellstr(strcat(inner_dir, listing_f(j).name));
                fnameshort = cellstr(strcat(person_id, '/', listing_f(j).name));
                
                filenames(inx, 1) = fnameshort;
                files{inx} = imread(char(fname));
                classes(inx, 1) = str2double(person_id);
                
                inx = inx + 1;
            end
            
            % Also read JSON annotation data:
            annotationsNew = awetcore_parse_annotations(person_id, annotations_f);
            if (~isequal(numel(annotations), 0))
                tabColNames = unique([annotations.Properties.VariableNames annotationsNew.Properties.VariableNames]);
                annotations = awetcore_equalize_tables(annotations, tabColNames);
                annotationsNew = awetcore_equalize_tables(annotationsNew, tabColNames);
                annotations = vertcat(annotations, annotationsNew);
            else
                annotations = annotationsNew;
            end
            
        end
    end
	db = table(files,classes,'RowNames',filenames,'VariableNames',{'image' 'class'});
    annotation_data = annotations;
end

function annotations = awetcore_parse_annotations(person_id, file_name)
    if (exist(file_name, 'file') == 2)
        data = loadjson(file_name);
        d = data.data;
        f = fieldnames(d);

        % First one is for init
        keys = {};
        for i = 1:numel(f)
            el = d.(f{i});
            elf = fieldnames(el);
            keys = [keys; elf];      
        end
        keys = sort(unique(keys));
        len = numel(f);
        wid = numel(keys);

        cellTab = cell(len, wid);
        filenames = cell(len, 1);

        % create table with columns of keys and length of len
        for i = 1:numel(f)
            el = d.(f{i});
            elf = fieldnames(el);
            vals = {};
            fnameshort = cellstr([person_id, '/', el.('file')]);
            filenames(i, 1) = fnameshort;
            for ik = 1:wid
                key = keys(ik);
                key = key{1};
                if (isfield(el, key))
                    val = el.(key);
                else
                    val = [];
                end        
                cellTab{i, ik} = val;
            end
        end

        annotations = cell2table(cellTab, 'VariableNames', keys, 'RowNames', filenames);
    else
        %awetcore_log('Annotations not present.', 2); 
        annotations = [];
    end
end

function modtab = awetcore_equalize_tables(tab, allColNames)
	tabColNames = tab.Properties.VariableNames;
	tabMissing = allColNames(~ismember(allColNames, tabColNames));

	if (size(tabMissing) > 0)
		tmp = cell(size(tab, 1), size(tabMissing, 2));
		tmpTable = cell2table(tmp, 'VariableNames', transpose(tabMissing));
		tab = [tab tmpTable];
	end
	modtab = tab;
end