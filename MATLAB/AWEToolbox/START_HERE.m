% This is the default execution file for the CVL ear toolbox version 1.0.
% If the following values are omitted (left empty) the defaults are used.

clear;clc;

% __ DATABASES ____________________________________________________________
% path = the path inside 'databases' folder to the database you want to use
% name = the label of the database to be shown in the output
% protocol = data divison type: 1 = random, 2 and 3 = AWE only (predifined division)
%
% if left blank THE DEFAULT VALUE IS
% databases = [
% 		struct('path', 'awe', 'name', 'AWE', 'protocol', 2)
% ];
databases = [];

% __ DB PREPROCESSOR ______________________________________________________
% if left blank THE DEFAULT VALUE IS
% preprocessor = struct('path', 'awet_basic', 'name', 'Basic preprocessing');
preprocessor = [];

% __ EXTRACTORS ___________________________________________________________
% path = the path inside 'extractors' folder to your .m files
% name = the label to be shown in the output
% distance = method of distance calculation. Chi requires compiled mex files - use cosine to work out-of-the-box
% bulk_postproces[opt] = whether to call postprocess once or always
% parameters = struct of parameters you can use in parameter tuning, e.g. struct('eta', [10, 100, 1000], 'aleph', [0.1, 0.2, 0.3]). Each combination is then accessible via awet.current_parameter_set
%
% if left blank THE DEFAULT VALUE IS
% extractors = [
% 		struct('path', 'awet_lbp',     'name', 'LBP',  'distance', 'chi', 'bulk_postprocess', false, 'parameters', struct())
% 		struct('path', 'awet_bsif',    'name', 'BSIF', 'distance', 'chi', 'bulk_postprocess', false, 'parameters', struct())
% 		struct('path', 'awet_lpq',     'name', 'LPQ', 'distance', 'chi', 'bulk_postprocess', false, 'parameters', struct())
% 		struct('path', 'awet_rilpq',   'name', 'RILPQ','distance', 'chi', 'bulk_postprocess', false, 'parameters', struct())    
% 		struct('path', 'awet_poem',    'name', 'POEM', 'distance', 'chi', 'bulk_postprocess', false, 'parameters', struct())
% 		struct('path', 'awet_hog',     'name', 'HOG',  'distance', 'chi', 'bulk_postprocess', false, 'parameters', struct())
% 		struct('path', 'awet_dsift',   'name', 'DSIFT','distance', 'chi', 'bulk_postprocess', false, 'parameters', struct())
% 		struct('path', 'awet_gabor',   'name', 'Gabor','distance', 'cosine', 'bulk_postprocess', false, 'parameters', struct())
% ];
extractors = [];

% __ MODELS ___________________________
% path = the path inside 'models' folder to your .m files
% name = the label to be shown in the output
% mode = it can be 'verification' or 'identification'
%
% if left blank THE DEFAULT VALUE IS
% models = [
%	struct('path', '', 'name', '', 'mode', 'none')
% ];
models = [];

% __ LOG LEVEL ____________________________________________________________
% Log level = 0 lowest, 2 highest level
%
% if left blank THE DEFAULT VALUE IS
% log_level = 1;
log_level = [];

% __ CROSS-VALIDATION _____________________________________________________
% method = 'Kfold' or 'HoldOut'
% factor = if Kfold then k (e.g. 3), if HoldOut then percentage for testset (e.g. 0.3)
% train_or_test = 'train', 'test' or 'both' (used only if train.txt and test.txt are present)
%
% ! IMPORTANT ! If DB contains train.txt and test.txt distribution files then
% method and factor parameters are ignored and 'train' parameter is used (and vice-versa)
%
% if left blank THE DEFAULT VALUE IS
% crossvalind = struct('method', 'Kfold', 'factor', 5, 'train_or_test', 'train');
%
crossvalind = [];

% __ OTHER ________________________________________________________________
protocol_type = []; % default = 1
norm_features = []; % default = 1
compressed_evaluation = []; % default = 1
use_phd = []; % PhD for CMC that is. ROC is always calculated using PhD anyways. Default = 1
norm_mode = [];% 1 or 2 etc., default = 2

% __ EXECUTE ______________________________________________________________
awet_init_and_run(databases, preprocessor, extractors, models, crossvalind, log_level, protocol_type, norm_features, compressed_evaluation, use_phd, norm_mode);