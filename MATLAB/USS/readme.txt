MATLAB Compiler

1. Prerequisites for Deployment 

. Verify the MATLAB Runtime is installed and ensure you    
  have installed version 9.0 (R2015b).   

. If the MATLAB Runtime is not installed, do the following:
  (1) enter
  
      >>mcrinstaller
      
      at MATLAB prompt. The MCRINSTALLER command displays the 
      location of the MATLAB Runtime installer.

  (2) run the MATLAB Runtime installer.

Or download the Windows 32-bit version of the MATLAB Runtime for R2015b 
from the MathWorks Web site by navigating to

   http://www.mathworks.com/products/compiler/mcr/index.html
   
   
For more information about the MATLAB Runtime and the MATLAB Runtime installer, see 
Package and Distribute in the MATLAB Compiler documentation  
in the MathWorks Documentation Center.    


NOTE: You will need administrator rights to run MCRInstaller. 


2. Files to Deploy and Package

Files to package for Standalone 
================================
-H_GLC.ctf (component technology file)
-H_GLC.exe
-MCRInstaller.exe 
   -if end users are unable to download the MATLAB Runtime using the above  
    link, include it when building your component by clicking 
    the "Runtime downloaded from web" link in the Deployment Tool
-This readme file 

3. Definitions

For information on deployment terminology, go to 
http://www.mathworks.com/help. Select MATLAB Compiler >   
Getting Started > About Application Deployment > 
Deployment Product Terms in the MathWorks Documentation 
Center.


4. Usage

From a command window, type:
  H_GLC.exe inputfilename
or
  H_GLC.exe inputfilename outputfilename


%% [Results] = H_GLC(filename)
% This function segments the sclera region in an eye image. 
% Input: 
% -infilename: the name of the file storing the image.
% -outfilename: the name of the output file image. if outfilename is [], the function compute the name of the output file automatically.
% 
% Output: 
% -Results: a structure with the following fields
%    + NormImage (double) : the gray level image obtained by applying color correction. 
%    + Binary (logical)   : bynary image after gray level clustering.
%    + Candidates (double): foreground regions that are candidates to be selected as sclera or part of it.
%    + Sclera (double)    : the binary image composed by pixels selected as  belonging to the sclera regions.







