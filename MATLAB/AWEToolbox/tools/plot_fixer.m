function plot_fixer()
%     plot_fixer_hists();
    plot_fixer_plots();
end

function plot_fixer_hists()
    folder = 'hists/';
    mods = {
        'P1_IITD1R-BSIF-chi';
        'P1_IITD2P-BSIF-chi';
        'P1_USTB1-BSIF-chi';
        'P1_USTB2-BSIF-chi';
        'P2_AWE-POEM-chi';
    };

    for i = 1:size(mods, 1)
        file = [folder, mods{i, 1}];
        mod_file_hist(file);
    end
end

function mod_file_hist(file)
    fg = openfig(file, 'visible');
    fsize = 38;
    
    set(gca, 'FontSize', fsize);
    set(findall(gcf, 'type', 'text'), 'fontSize', fsize);
    
    saveas(fg, [file, '.pdf']);
end

function plot_fixer_plots()
    folder = 'plots/';
    mods = {
        'P1_IITD1R-CMC_crop'     1 3;
        'P1_IITD1R-ROClog_crop'  1 2;
        'P1_IITD2P-CMC_crop'     1 3;
        'P1_IITD2P-ROClog_crop'  1 2;
        'P1_USTB1-CMC_crop'      1 3;
        'P1_USTB1-ROClog_crop'   1 1;
        'P1_USTB2-CMC_crop'      1 3;
        'P1_USTB2-ROClog_crop'   1 1;
        'P2_AWE-CMC_crop'        0 3;
        'P2_AWE-ROClog_crop'     0 2;
        'P3_AWE-CMC_crop'        0 3;
        'P3_AWE-ROClog_crop'     0 2;
    };
    
    for i = 1:size(mods, 1)
        file = [folder, mods{i, 1}];
        sizeClass = mods{i, 2};
        roclog = mods{i, 3};
        mod_file_plot(file, sizeClass, roclog);
    end
end

function mod_file_plot(file, sizeClass, roclog)
    %fg = openfig(file, 'visible');
    fg = openfig(file);
    ax = gca;

    if (sizeClass == 1) % small
        fsize = 28;
        thickness = 6;
    else % large
        fsize = 20;
        thickness = 3;
    end
    
    % larger font
    set(gca, 'FontSize', fsize);
    set(findall(gcf, 'type', 'text'), 'fontSize', fsize);
    
    if (roclog == 1)
        ax.XLim = [1e-2 1];
        set(gca,'XTick',[1e-2 1e-1 1] );
        ax.XScale = 'log';
        xlabel('False Acc. Rate')
        ylabel('True Acc. Rate')
    elseif (roclog == 2)
        set(gca,'XTick',[1e-3 1e-2 1e-1 1] );
        ax.XScale = 'log';
        xlabel('False Acc. Rate')
        ylabel('True Acc. Rate')
    elseif (roclog == 3)
        xlabel('Rank')
        ylabel('Recognition Rate')
    end
    
    box on;
    
    % semi-transparent legend
    leg = legend;
    set(leg.BoxFace, 'ColorType', 'truecoloralpha', 'ColorData', uint8(255*[1;1;1;.6]));
    
    hline = findobj(gcf, 'type', 'line');
    set(hline, 'LineWidth', thickness);

    saveas(fg, [file, '.pdf']);
end