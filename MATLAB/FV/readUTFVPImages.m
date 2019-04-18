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
% This function reads the UTFVP images and stores them in a MATLAB cell
% array with the dimensions (60x6x4) as greyscale images
%
% Parameters:
%  path     - The input path where the images are located
%  fileExt  - The image file extension (default is *.png)
%  varargin - Optional input arguments (not used at the moment)
%
% Returns:
%  images - 3D cell array containing the greyscale images (60x6x4)
% *************************************************************************
function images = readUTFVPImages(path, fileExt, varargin)
    % Prompt the user to choose the image directory
    if nargin >= 1
        fileFolder = path;
    else
        fileFolder = uigetdir();
    end
    if nargin < 2
        fileExt = '*.png';
    end
	
    
    fprintf('Read all images inside: %s\n', fileFolder);
    fileNames = getAllFiles(fileFolder, fileExt);
    numberOfImageFiles = numel(fileNames);
    fprintf('\tNumber of Subdirectories: %d\n', size(fileNames, 1));
    fprintf('\tNumber of Images: %d\n', numberOfImageFiles);
    fprintf('...\n');
    
    if numberOfImageFiles ~= 1440
        fprintf(2, 'ERROR: Incorrect number %d of images found in UTFVP path. Should be 1440 images.\n', numberOfImageFiles);
        images = [];
        return;
    end

    % Read all the UTFVP images and store then in a matrix according to
    % Subject ID, Finger ID, Image ID
    % Create the cell array for storing the images
    images = cell(60,6,4);   % For UTFVP images
    % Read the images
    for p = 1 : numberOfImageFiles
        [~, filename, ext] = fileparts(fileNames{p});
        currentFileName = [filename, ext];
        inputIDs = regexp(currentFileName, '^(?<subjectID>\d+)_(?<fingerID>\d+)_(?<imageID>\d+)_.*$', 'names');
        subjectID = str2double(inputIDs.subjectID);
        fingerID = str2double(inputIDs.fingerID);
        imageID = str2double(inputIDs.imageID);
        % Read image
		currentImage = imread(fileNames{p});
        % Convert to grayscale if color image
        if size(currentImage, 3) == 3
            currentImage = rgb2gray(currentImage);
        end
        % Store the image in the cell array
        images{subjectID, fingerID, imageID} = currentImage;
    end
    
    fprintf('Reading images finished.\n');
end


% Gets all files in the given directory dirName including subdirectories matching
% the regular expression regex
function fileList = getAllFiles(dirName, regex)
    dirData = dir(dirName);      %# Get the data for the current directory
    dirIndex = [dirData.isdir];  %# Find the index for directories
    fileList = {dirData(~dirIndex).name}';  %'# Get a list of the files
    %# only use fiels where regex holds
    %fileList = regexpi({dirData.name},regex,'match');
    if ~isempty(fileList)
        filePattern = fullfile(dirName, regex);
        dirOutput = dir(filePattern);
        fileList = cellfun(@(x) fullfile(dirName, filesep , x), ...
            {dirOutput.name}, 'UniformOutput', false);
    end
    subDirs = {dirData(dirIndex).name};  %# Get a list of the subdirectories
    validIndex = ~ismember(subDirs,{'.','..'});  %# Find index of subdirectories
                                               %#   that are not '.' or '..'
    for iDir = find(validIndex)                  %# Loop over valid subdirectories
        nextDir = fullfile(dirName,subDirs{iDir});    %# Get the subdirectory path
        currentFileList = getAllFiles(nextDir,regex);  %# Recursively call getAllFiles
        if ~isempty(currentFileList)
            if isempty(fileList)
                fileList = currentFileList;  
            else
                fileList = [fileList, currentFileList];
            end
        end
    end
end