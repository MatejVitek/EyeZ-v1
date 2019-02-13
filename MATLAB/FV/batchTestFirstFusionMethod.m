% Test file for batch testing the first fusion method

% For batch testing this has to be done only once
setup_Fusion;

featureTypes = {'MC', 'PC', 'GB', 'WLD', 'IUWT', 'RLT'};
fusionStrategies = {'MV', 'Av', 'STAPLE', 'STAPLE_prob', 'STAPLER', ...
                    'STAPLER_prob', 'COLLATE', 'COLLATE_prob'};
            
            
for f=1:numel(featureTypes)
    for s=1:numel(fusionStrategies)
        try
            runFusionFirstMethod(featureTypes{f}, fusionStrategies{s}, false);
        catch ME1
            fprintf(2, 'ERROR: Unable to run the fusion for %s, %s\n', featureTypes{f}, fusionStrategies{s});
        end     
    end
end