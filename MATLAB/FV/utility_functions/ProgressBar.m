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
% This class provides a progress bar functionality to display the current
% progress and also the remaining time using the MATLAB waitbar. It is a
% static (singleton) class providing only an update function.
%
% Parameters:
%  value  - The current progress (0 to 1). Special values are possible:
%           new ... Creates a new progress bar. Only there the title is set
%           close ... Closes the current progress bar.
%  text   - The text to display inside the progress bar (status text).
%  title  - The title of the progress bar window. Used to give additional
%           information.
% *************************************************************************
classdef ProgressBar
    %PROGRESSBAR Displays a progress bar using waitbar
    %   Detailed explanation goes here
    
    properties
    end
    
    methods(Static)
        function update(value, text, title)
            persistent h;
            persistent displayText;
			
            if ~feature('ShowFigureWindows')    % For now: just don't diplay, better would be text display instead
                return;
            end
			
            if (strcmpi('new', value))
                if nargin == 3
                    h = waitbar(0, text, 'Name', title);
                else
                    h = waitbar(0, text);
                end
                displayText = text;
            elseif (strcmpi('close', value))
                if (ishandle(h))
                    close(h);
                end
            else
                if (~ishandle(h))  % If waitbar was closed create new one
                    h = waitbar(value, displayText);
                end
                if nargin < 2
                    h = waitbar(value);
                else
                    h = waitbar(value, h, [displayText text]);
                end
            end
        end
    end
end
