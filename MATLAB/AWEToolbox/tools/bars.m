fontSize = 24;
barColor = [0.3 0.3 0.3];
data = struct();

data(1).name = 'accessories';
data(1).labels = {'None' 'Earrings' 'Other'};
data(1).rightAlignment = [1 0 0];
data(1).numbers = [913 78 9];

data(2).name = 'side';
data(2).labels = {'Left' 'Right'};
data(2).rightAlignment = [0 0];
data(2).numbers = [520 480];

data(3).name = 'pitch';
data(3).labels = {'Up ++' 'Up +' 'Neutral' 'Down +' 'Down ++'};
data(3).rightAlignment = [0 0 0 0 0];
data(3).numbers = [23 156 543 250 28];

data(4).name = 'roll';
data(4).labels = {'To Right ++' 'To Right +' 'Neutral' 'To Left +' 'To Left ++'};
data(4).rightAlignment = [0 0 0 0 0];
data(4).numbers = [24 181 625 151 19];

data(5).name = 'yaw';
data(5).labels = {'Frontal Left' 'Middle Left' 'Profile Left' 'Profile Right' 'Middle Right' 'Frontal Right'};
data(5).rightAlignment = [0 0 0 0 0 0];
data(5).numbers = [143 313 64 86 249 145];

data(6).name = 'occlusion';
data(6).labels = {'None' 'Mild' 'Severe'};
data(6).rightAlignment = [0 0 0];
data(6).numbers = [651 275 74];

data(7).name = 'gender';
data(7).labels = {'Female', 'Male'};
data(7).rightAlignment = [0 1];
data(7).numbers = [9 91];

data(8).name = 'ethnicity';
data(8).labels = {'White' 'Asian' 'South Asian' 'Black' 'Middle Eastern' 'South American' 'Other'};
data(8).rightAlignment = [0 0 0 0 0 0 0];
data(8).numbers = [61 18 3 11 3 3 1];

for i = 1:length(data)
    h = figure;
    name = data(i).name;
    labels = fliplr(data(i).labels);
    rightAlignment = fliplr(data(i).rightAlignment);
    numbers = fliplr(data(i).numbers);
    disp(sum(numbers));
    numbers = 100 * numbers / sum(numbers);
   
    barWidth = 0.7;
    
    barh(numbers, 'BarWidth', barWidth, 'EdgeColor', 'none', 'FaceColor', barColor);
    %set(gca,'YTickLabel', labels)
    xlim([0 100]);
    hold on
    
    % plot tufte lines
    lns = 0:20:100;
    lns = lns(2:end);
    for j = 1:length(lns)
        line([lns(j) lns(j)], [0 length(numbers) + barWidth], 'color', 'w', 'LineWidth', 3);
    end
    
    % set steps
    set(gca, 'xtick', [0 lns]);
    set(gca, 'XTickLabel', {'[%]', '20', '40', '60', '80', '100'});
    set(gca, 'ytick', []);
    set(gca, 'box', 'off')
    
    % set font size
    set(gca, 'FontSize', fontSize);
    set(findall(gcf, 'type', 'text'), 'fontSize', fontSize);
    
    % display labels
    yOffset = 2;
    for k = 1:length(numbers)
        num = numbers(k);
        lab = {num2str(round(num)), '% ', labels{k}};
        
        if (rightAlignment(k) == 0)
            horzAlign = 'left';
            fColor = [0 0 0];
            bColor = 'none';
            
            num = num + yOffset;
        else 
            horzAlign = 'right';
            fColor = [1 1 1];
            bColor = barColor;
            
            num = num - yOffset;
        end
        text(num, k, strjoin(lab, ''), 'fontSize', fontSize, 'margin', 0.1, 'HorizontalAlignment', horzAlign, 'Color', fColor, 'BackgroundColor', bColor);
    end
    
    % save it
    set(h, 'Units', 'Inches');
    pos = get(h, 'Position');
    set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    print(h, ['awed_plots/', name], '-dpdf', '-r0');
    
    savefig(['awed_plots/', name]);
end