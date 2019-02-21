function awetcore_plot_hist(client_distances, impostor_distances)
    global awet;

    suffix = [awet.current_extractor.name, '-', awet.current_extractor.distance];

    [~, filename_png, filename_fig, filename_pdf] = awetcore_check_dir(suffix, '/histograms', 'png', 'fig', 'pdf');
    
    hold on
    set(0,'DefaultFigureVisible','off')
    
    fg = figure;

    MODE = 2; % 1 = bins, 2 = line, 3 = normalized line
    NBINS = 40;
    
    if (MODE == 1)
        histogram(client_distances, NBINS)
        h = findobj(gca, 'Type', 'patch');
        set(h, 'FaceColor', [0.82 0.03 0.03], 'facealpha', 0.8)
        hold on;
        histogram(impostor_distances, NBINS)
        h1 = findobj(gca,'Type','patch');
        set(h1, 'FaceColor', [0.15,0.54,0.82], 'facealpha', 0.8);
    elseif (MODE == 2)
        h1 = histogram(client_distances);
        [x1, y1] = getCurveFromHist(h1);
        hold on;
        h2 = histogram(impostor_distances);
        [x2, y2] = getCurveFromHist(h2);
        
        fg = figure;
        plot(x1, y1, 'LineWidth', 6,'LineStyle', ':', 'color', [0 0 0]);
        hold on;
        plot(x2, y2, 'LineWidth', 6,'LineStyle', '--', 'color', [0.6 0.6 0.6]);
        
        xmin = min(min(x1), min(x2));
        xmax = max(max(x1), max(x2));
        ymin = min(min(y1), min(y2));
        ymax = max(max(y1), max(y2));
        
        ax = gca;
        ax.XLim = [xmin xmax];
        ax.YLim = [ymin ymax];
        
        set(gca, 'FontSize', 28);
        set(findall(gcf, 'type', 'text'), 'fontSize', 28);
        set(gca,'YTickMode', 'auto', 'YTickLabel', {'0','1'}, 'YTick', [ymin ymax]);
        %set(gca,'XTickMode', 'auto', 'XTickLabel', {'0','1'}, 'XTick', [xmin xmax]);
        
    else

    end    
    
    fsize = 38;    
    set(gca, 'FontSize', fsize);
    set(findall(gcf, 'type', 'text'), 'fontSize', fsize);
    
    leg = legend('clients', 'impostors', 'Location', 'northwest');
    % semi-transparent legend
    set(leg.BoxFace, 'ColorType', 'truecoloralpha', 'ColorData', uint8(255*[1;1;1;.8]));
    
    %awetcore_remove_margins();
    
    saveas(fg, [filename_png, awet.parameter_path, '.png']);
    savefig(fg, [filename_fig, awet.parameter_path, '.fig']);
    saveas(fg, [filename_pdf, awet.parameter_path, '.pdf']);
    
    hold off
    set(0,'DefaultFigureVisible','on')
    

end

function [x, y] = getCurveFromHist(h)
    stepH = h.BinWidth;
    startH = h.BinLimits(1) + (stepH/2);
    endH = h.BinLimits(2);
    x = startH:stepH:endH;
  
    v = ver;
    if any(strcmp({v.Name}, 'Curve Fitting Toolbox'))
        y = smooth(h.Values);
    else
        y = h.Values';
    end
    y = y / trapz(x, y);
    
    x = [startH-stepH x endH+stepH];
    y = [0; y; 0];
end

