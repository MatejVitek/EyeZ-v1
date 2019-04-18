% *************************************************************************
% This file is part of the feature level fusion framework for finger vein
% recognition (MATLAB implementation). 
%
% Reference:
% Advanced Variants of Feature Level Fusion for Finger Vein Recognition 
% C. Kauba, E. Piciucco, E. Maiorana, P. Campisi and A. Uhl 
% In Proceedings of the International Conference of the Biometrics Special 
% Interest Group (BIOSIG'16), pp. 1-12, Darmstadt, Germany, Sept. 21 - 23
%
% Authors: Christof Kauba <ckauba@cosy.sbg.ac.at> and 
%          Emanuela Piciucco <emanuela.piciucco@stud.uniroma3.it>
% Date:    31th August 2016
% License: Simplified BSD License
%
% 
% Description:
% This function updates the current status progress bar (the bar itself and
% the status text with the remaining time.
%
% Parameters:
%  current      - Number of completed items (integer)
%  total        - Number of total items to process
%  elapsedTime  - Current elapsed time in s. Can be obtained using the tic
%                 and toc commands.
%
% Returns:
%  percentCompleted - The number of completed items in percentage of total
% *************************************************************************
function percentCompleted = updateStatus(current, total, elapsedTime)
%UPDATESTATUS Updates the current status text
%   Detailed explanation goes here
    currentCompleted = current/total;
    estimatedTime = elapsedTime/currentCompleted - elapsedTime;
    percentCompleted = currentCompleted*100;
    ProgressBar.update(currentCompleted, sprintf(' (ETA: %s)', getTimeString(estimatedTime)));
end

function timeString = getTimeString(time)
%GETTIMESTRING Get a string with time in s/min/h/days
%   Detailed explanation goes here
    estimatedTime = time;
    timeUnit = 's';
    if estimatedTime > 60
        estimatedTime = estimatedTime / 60;
        timeUnit = 'min';
        if estimatedTime > 60
            estimatedTime = estimatedTime / 60;
            timeUnit = 'h';
            if estimatedTime > 24
                estimatedTime = estimatedTime /24;
                timeUnit = 'd';
            end
        end
    end
    timeString = sprintf('%.2f %s', estimatedTime, timeUnit);
end
