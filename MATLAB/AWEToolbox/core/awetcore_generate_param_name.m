function name = awetcore_generate_param_name(parameterSet)    
    name = '';
    if ~isempty(parameterSet)
        arrParamName = fieldnames(parameterSet);
        if (~isempty(arrParamName))
            for i = 1:length(arrParamName)
                paramName = arrParamName(i);
                name = strcat(name,'-',paramName{:},'=',num2str(parameterSet.(paramName{:})));
            end
            %name = name(2:end);    
        end
    end
end