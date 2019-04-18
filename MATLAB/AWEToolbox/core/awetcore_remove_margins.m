function awetcore_remove_margins(height)
% Remove Matlab margins when saving figures
% Thanks to: http://tipstrickshowtos.blogspot.si/2010/08/how-to-get-rid-of-white-margin-in.html

    if ~exist('height','var')
        ti = get(gca,'TightInset');

        set(gca,'Position',[ti(1) ti(2) 1-ti(3)-ti(1) 1-ti(4)-ti(2)]);

        set(gca,'units','centimeters')
        pos = get(gca,'Position');
        ti = get(gca,'TightInset');

        height = pos(4);
        
        set(gcf, 'PaperUnits','centimeters');
        set(gcf, 'PaperSize', [pos(3)+ti(1)+ti(3) height+ti(2)+ti(4)]);
        set(gcf, 'PaperPositionMode', 'manual');
        set(gcf, 'PaperPosition',[0 0 pos(3)+ti(1)+ti(3) height+ti(2)+ti(4)]);
    else
        set(gcf,'PaperUnits','centimeters')
        xSize = 30; ySize = 5;
        xLeft = (20-xSize)/2; yTop = (30-ySize)/2;
        set(gcf, 'PaperPositionMode', 'manual');
        set(gcf,'PaperPosition',[xLeft yTop xSize ySize])
        set(gcf,'Position',[300 600 xSize*50 ySize*50])
    end
end

