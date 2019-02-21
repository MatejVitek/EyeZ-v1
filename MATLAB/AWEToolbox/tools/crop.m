root = '../databases/wputedb_cropped/';
file_endings = {'png', 'jpg', 'bmp'};
len = 0;

% first run for allocation
listing = dir(root);
for i = 1:numel(listing)
    if (listing(i).isdir && ~strcmp(listing(i).name, '.') && ~strcmp(listing(i).name, '..'))
        f = dir(strcat(root, listing(i).name, '/*'));
        f = f(~cellfun(@isempty,regexpi({f.name}, ['.*(', strjoin(file_endings, '|'), ')'])));
        len = len + numel(f);
    end
end

for i = 1:numel(listing)
    if (listing(i).isdir && ~strcmp(listing(i).name, '.') && ~strcmp(listing(i).name, '..'))
        person_id = listing(i).name;
        inner_dir = strcat(root, person_id, '/');
        listing_f = dir(strcat(inner_dir, '*'));
        listing_f = listing_f(~cellfun(@isempty,regexpi({listing_f.name}, ['.*(', strjoin(file_endings, '|'), ')'])));

        for j = 1:numel(listing_f)
            fname = strcat(inner_dir, listing_f(j).name);

            I = imread(fname);
            
            h = size(I,1);
            w = size(I,2);
            hN = 0.1 * h;
            wN = 0.2 * w;
            
            I2 = imcrop(I, [wN hN (w - (2*wN)) (h - (2*hN))]);
            
            %figure;
            %imshow(I)
            %figure
            %imshow(I2);
            imwrite(I2, fname);
        end
    end
    disp([num2str(i), ' DONE']);
end
disp('FINISHED!!');