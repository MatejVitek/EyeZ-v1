% Test file for batch testing the second fusion method

% For batch testing this has to be done only once
setup_Fusion;

selectedFeatures = {6, 1, 2, 4, 5, 3, 7};
fusionStrategies = {'MV', 'Av', 'STAPLE', 'STAPLE_prob', 'STAPLER', ...
                    'STAPLER_prob', 'COLLATE', 'COLLATE_prob'};
            
            
for f=1:numel(selectedFeatures)
    for s=1:numel(fusionStrategies)
        try
            runFusionSecondMethod(selectedFeatures{f}, fusionStrategies{s});
        catch ME1
            fprintf(2, 'ERROR: Unable to run the fusion for %d %s\n', selectedFeatures{f}, fusionStrategies{s}, false);
        end     
    end
end