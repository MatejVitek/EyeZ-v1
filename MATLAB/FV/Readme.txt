Finger Vein Feature Level Fusion Framework (MATLAB Impementation)
=================================================================

This package contains all the software to run the experiments of the following paper:
Advanced Variants of Feature Level Fusion for Finger Vein Recognition 
C. Kauba, E. Piciucco, E. Maiorana, P. Campisi and A. Uhl 
In Proceedings of the International Conference of the Biometrics Special 
Interest Group (BIOSIG'16), pp. 1-12, Darmstadt, Germany, September 21 - 23

Authors: Christof Kauba <ckauba@cosy.sbg.ac.at> and 
         Emanuela Piciucco <emanuela.piciucco@stud.uniroma3.it>
Date:    31th August 2016
License: Simplified BSD License


General information
===================

The software is written in MATLAB and was tested with MATLAB 2013b. It should run an all MATLAB versions higher or equal to 2013b. 

The following feature extraction methods are included:
- Maximum Curvature
- Repeated Line Tracking
- Wide Line Detector
- Principal Curvature
- Gabor Filter
- IUWT

Two different types of fusion are implemented:
- Single feature extractor only: Fusing the outputs of a single feature extractor while varying its parameters
- Multiple feature extractors: Fusing the outputs of (all) different feature extractors

The following fusion schemes are included:
- Majority Voting
- Weighted Average
- STAPLE
- STAPLER
- COLLATE

Package information
-------------------

The main directory contains the setup and the main functions for the two different types of fusion, including the matching/score calculation function. The package further contains the following subdirectories:
- data: Here the UTFVP images should be placed and all the feature, score and results files can be found
- functions_eer_evaluation: Functions from the Biosecure Tool to determine EER and ROC curves (modified)
- functions_feature_extraction: Implementations of the different feature extractors
- functions_fusion: Implementations of the fusion methods (MV, Av and wrapper for STAPLE/STAPLER/COLLATE)
- functions_ton: Functions from B.T. Ton for feature extractors and matching
- iuwt_vessels_lib: Path where the ARIA Vessels Lib should be put into
- masi-fusion: Path where the MASI Fusion framework should be placed
- utility_functions: Some utility functions, e.g. a progress bar class

As some feature extractors / fusion schemes depend on external software, a few dependencies have to be resolved before the software can be used:

External dependencies
---------------------

- Dataset: the UTFVP dataset was used during the experiments of the paper. The UTFVP image files have to be put in the "data/utfvp" directory.
- ARIA Vessel Library: Needed the IUWT feature extraction. It can be downloaded from https://sourceforge.net/projects/aria-vessels/. The files should be put in the "iuwt_vessels_lib" directory.
- MASI Fusion Framework: Implements STAPLE/STAPLER/COLLATE fusion schemes besides other things. Can be downloaded from https://www.nitrc.org/projects/masi-fusion/ and should be put into the "masi-fusion" directory or any other directory to which the path is set in "setup_Fusion.m" then.

Other dependencies
------------------

The following methods have not been implemented by ourselves but are included in the framework sources:

For the Maximum Curvature, Repeated Line Tracking, Wide Line Detector and Principal Curvature feature extraction as well as for the finger boundary detection and the finger normalisation an implementation of B.T. Ton was utilised which is publicly availabale through MATLAB Central (http://www.mathworks.nl/matlabcentral/fileexchange/authors/57311).
For matching an implementation of B.T. Ton of the method proposed by Miura et al. which can also be downloaded via MATLAB Central (http://www.mathworks.com/matlabcentral/fileexchange/35716-miura-et-al-vein-extraction-methods) was used. All these implementations can be found in the "functions_ton" directory.

For the EER determination the routine of the Biosecure Tool was utilisied which can be found here: http://svnext.it-sudparis.eu/svnview2-eph/ref_syst/Tools/PerformanceEvaluation/. 

For the adaptive thresholding method (adaptivethreshold.m) an implementation by Guanglei Xiong is used which can be downloaded via MATLAB Central http://de.mathworks.com/matlabcentral/fileexchange/8647-local-adaptive-thresholding/content/adaptivethreshold/adaptivethreshold.m.
For the 2-D filtering using Gaussian masks (ut_gauss.m) a publicly available implementation by F. van der Heijden that can be downloaded via MATLAB Central:  https://de.mathworks.com/matlabcentral/fileexchange/25449-image-edge-enhancing-coherence-filter-toolbox/content/functions2D/ut_gauss.m is used.


Usage Instructions
==================

Prequisites
-----------

After putting the necessary external stuff in the corresponding directories, the correct paths should be set up in the file "setup_Fusion.m". Especially the path to the MASI and the ARIA library.
All the settings (fusion, feature extraction, matching, etc.) can be changed in the file setup_Settings.m. There a settings struct is created which contains substructs for each processing step. If you need to change some parameters this should be done here.

Running the experiments
-----------------------

The fusion process works in the following steps:

1.) The UTFVP images are read, finger boundaries are detected, preprocessing is done and the ROI and preprocessed images are stored.
2.) Features are extracted using one or several of the six feature extractors and stored.
3.) The chosen fusion method is applied and the fused features are stored.
4.) Matching is performed and the matching scores are stored.
5.) The EER and ROC plots based on the matching scores are calculated.

The first type of fusion (only a single feature extractor) can be run using the "runFusionFirstMethod.m" MATLAB file. Either call the "runFusionFirstMethod" function with its two parameters, the first is the feature extractor and the second is the fusion method or call it without any parameters then it asks for the two values.

The second type of fusion (using several feature extractors) can be run using the "runFusionSecondMethod.m" MATLAB file. If it is called without any parameter, it asks the user for the feature extractor combination and the fusion method to use. Optionally these two parameters can be given when the function is called.

In both files some additional parameters can be specified (true/false):
- override_results 	Override existing results or not
- save_features 	Save the feature files after the feature extraction step (step 2)
- save_fused_features 	Save the feature files after fusion (step 3)
- do_matching 		Perform the matching and score calculation (step 4)
- save_scores 		Save the scores after matching is finished (step 4)
- do_eer_calculation 	Calculate the EER (equal error rate) (step 5)
- save_results		Save the EER results (step 5)

If some of the saved files are already present and "override_results" is set to false, the files are loaded instead of running the corresponding step once more.