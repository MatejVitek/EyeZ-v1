function awetcore_add_plot_and_draw_all(type, x, y, y0, y1, xlabel, ylabel, legend_label)
    global awet;
    %% Results to temp so that graphs get redrawn at each iteration
    dbname = ['P', num2str(awet.current_database.protocol), '_', awet.current_database.name];
    plen = size(awet.extractors, 1);
    if ~isfield(awet.plots, dbname) || ~isfield(awet.plots.(dbname), type)
        awet.plots.(dbname).(type) = struct();
        awet.plots.(dbname).(type).i = 1;
        awet.plots.(dbname).(type).legend = cell(plen, 1);
        awet.plots.(dbname).(type).x = cell(plen, 1);
        awet.plots.(dbname).(type).y = cell(plen, 1);
    end
    
    % If previously set, it will be overriden
    awet.plots.(dbname).(type).xlabel = xlabel;
    awet.plots.(dbname).(type).ylabel = ylabel;
    
    % Add data to plots data
    pinx = awet.plots.(dbname).(type).i;
    awet.plots.(dbname).(type).x{pinx} = x;
    awet.plots.(dbname).(type).y{pinx} = y;
    awet.plots.(dbname).(type).y0{pinx} = y0;
    awet.plots.(dbname).(type).y1{pinx} = y1;
    awet.plots.(dbname).(type).legend{pinx} = legend_label;
    awet.plots.(dbname).(type).i = pinx + 1;
    
    % Draw all graphs together
    plot_all_current_of_type(type);
end

function plot_all_current_of_type(type)
    global awet;
    
    PLOT_STD = true;
    
    dbname = ['P', num2str(awet.current_database.protocol), '_', awet.current_database.name];
    limit = awet.plots.(dbname).(type).i - 1;
    
    hold off
    set(0,'DefaultFigureVisible','off')
    fg = figure;
    hold on
    grid on
    
    plt_hands = zeros(limit, 1);
    min_x_els = realmax;
    yAxisMin = realmax;
    for i = 1:limit
        x = awet.plots.(dbname).(type).x{i};
        y = awet.plots.(dbname).(type).y{i};
        y0 = awet.plots.(dbname).(type).y0{i};
        y1 = awet.plots.(dbname).(type).y1{i};
        color = awet.plot_colors(mod(i-1,size(awet.plot_colors, 1))+1, :);

%         if (strcmp(type, 'ROClog') == 1 || strcmp(type, 'ROC') == 1)
%             % Add 0,0 so that lines start at 0,0
%             x = [0 x];
%             y = [0 y];
%             y0 = [0 y0];
%             y1 = [0 y1];
%         end
        
        if (strcmp(type, 'ROClog') == 1 || strcmp(type, 'ROClog_crop') == 1)
            % Logarithmic x-axis
            set(gca, 'XScale', 'log')
            plt_hands(i) = semilogx(x, y, 'color', color, 'LineWidth', 3);
            if (PLOT_STD)
                semilogx(x, y0, 'color', color, 'lineStyle', ':', 'LineWidth', 2);
                semilogx(x, y1, 'color', color, 'lineStyle', ':', 'LineWidth', 2);
            end
        elseif (strcmp(type, 'ROC') == 1 || strcmp(type, 'CMC') == 1 || strcmp(type, 'ROC_crop') == 1 || strcmp(type, 'CMC_crop') == 1)
            % Linear x-axis
            plt_hands(i) = plot(x, y, 'color', color, 'LineWidth', 3);
            if (PLOT_STD)
                plot(x, y0, 'color', color, 'lineStyle', ':', 'LineWidth', 2);
                plot(x, y1, 'color', color, 'lineStyle', ':', 'LineWidth', 2);
            end
            if (strcmp(type, 'ROC') == 1)
                plot([0 1], [1 0], 'color', [0.1 0.1 0.1], 'lineStyle', '--');
                plot([0 0], [1 1], 'color', [0.2 0.2 0.2], 'lineStyle', ':');
            elseif (strcmp(type, 'CMC') == 1 && min_x_els < 10)
                set(gca, 'xtick', 1:(numel(x)));
            end
        else
            disp('plotting default');
            plot_ROC_PhD(y, x, 'b',4);
            xlabel('False Acceptance Rate')
            ylabel('False Rejection Rate')
        end
        curr_els = numel(x);
        if (curr_els < min_x_els)
            min_x_els = curr_els;
        end
        
        %ymin = min(min([y, y0, y1]));
        ymin = min(y);
        if (ymin < yAxisMin)
            yAxisMin = ymin;
        end
    end
    if strcmp(type, 'ROCPhD') == 0
        xlabel(awet.plots.(dbname).(type).xlabel);
        ylabel(awet.plots.(dbname).(type).ylabel);
        leg = legend(plt_hands, awet.plots.(dbname).(type).legend{1:limit}, 'Location', 'southeast');
        set(leg.BoxFace, 'ColorType', 'truecoloralpha', 'ColorData', uint8(255*[1;1;1;.6]));
    end
    
    if (strcmp(type, 'ROC') == 1 || strcmp(type, 'CMC') == 1)
        % no cropping
        yAxisMin = 0;
    else
        diffYAxis = 1 - yAxisMin;
        yAxisMin = yAxisMin - (diffYAxis/50);
    end
    
    if yAxisMin < 0.1
        % set y to 0.x floor
        yAxisMin = 0;
    end
    
    if (strcmp(type, 'ROC') == 1)
        axis([0 1 yAxisMin 1]);
    elseif (strcmp(type, 'ROClog') == 1)
        axis([1e-3 1 yAxisMin 1]);
        %axis([9*(1e-4) 1 yAxisMin 1]);
        %axis([0 0.1 0.9 1]);
    elseif (strcmp(type, 'CMC') == 1 || strcmp(type, 'CMC_crop') == 1)
        axis([1 min_x_els yAxisMin 1])
    end
    
	[filename_plain, filename_png, filename_fig, filename_pdf] = awetcore_check_dir(type, '/plots', 'png', 'fig', 'pdf');
    
    %if (strcmp(type, 'ROC_crop') == 1 || strcmp(type, 'ROClog_crop') == 1 || strcmp(type, 'CMC_crop') == 1)
    %    awetcore_remove_margins(3);
   % else
        %awetcore_remove_margins();
    %end
    
    set(gca,'FontSize',20);
    set(findall(gcf,'type','text'),'fontSize',20);
    
    saveas(fg, [filename_png, '.png']);
    savefig(fg, [filename_fig, '.fig']);
    saveas(fg, [filename_pdf, '.pdf']);
    
    hold off
    set(0,'DefaultFigureVisible','on')
end