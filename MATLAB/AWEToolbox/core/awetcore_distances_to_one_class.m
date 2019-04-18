function [R, T, classes_x, classes_y] = awetcore_distances_to_one_class(outputs, targets, classes)
    SENTINEL = realmax;
    unique_classes = unique(classes);
    classes_num = size(unique_classes, 1);
    R = zeros(size(outputs, 1), classes_num);
    T = zeros(size(outputs, 1), classes_num);
    outputs(logical(eye(size(outputs)))) = SENTINEL;
    % set the same to -2
%     clen = length(classes);
%     for c = 1:clen
%         line_c = classes(c);
%         for v = 1:clen
%             cell_c = classes(v);
%             if (line_c == cell_c)
%                 outputs(c, v) = SENTINEL;
%             end
%         end
%     end
    
    % Sprehajaj se po vrsticah outputa
        % v vsaki vrstici se sprehodi po razredih
            % za vsak razred poisci max vrednost in to shrani v polje v Rju
            % v T shranis glede na to, ce je classes(st. vrstice v outputu)
            % == current razred, das 1, ce ne das 0.
    for i = 1:size(outputs, 1)
        line = outputs(i, :);
        line_class = classes(i);
        for c = 1:classes_num
            class = unique_classes(c);
            min_val = min(line(classes==class));
            R(i, c) = min_val;
            if class == line_class
                T(i, c) = 1;
                % also set r to 1 if needed so that we don't have -2
                if R(i, c) == SENTINEL
                    awetcore_log('FATAL ERROR - only one sample of class available! At least two are required. Shuting down ...', 0);
                    return;
                end
            end
        end
    end
    classes_x = unique_classes;
    classes_y = classes;
    
%     for i = 1:classes_num
%         class = all_classes(i);
%         inx = find(classes==class);
%         inxR = inx;
%         inxC = inx;
%         r = R(inx, inx);
%         
%         if (max_or_mean_mode == 0)
%             [max_val max_loc] = max(r(:));
%         elseif (max_or_mean_mode == 1)
%             [max_val max_loc] = min(abs(r(:) - mean(r(:))));
%         elseif (max_or_mean_mode == 2)
%             [max_val max_loc] = min(r(:));
%         end
%         [max_loc_row max_loc_col] = ind2sub(size(r), max_loc);
%         inxR(max_loc_row) = [];
%         inxC(max_loc_col) = [];
%         
%         R(inxR, :) = [];
%         R(:, inxC) = [];
%         classes(inx(2:end)) = [];
%     end
%     
%     Rs = R;
end

