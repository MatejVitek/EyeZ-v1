function awetcore_inface_install()
    if (exist('single_scale_retinex','file'))
        awetcore_log('INFace is installed.\n', 2);
    else
        awetcore_log('INFace NOT installed. Attempting install ...\n', 2);
        
        addpath(genpath('libraries/INface_tool'));
        
        awetcore_log('INFace installed.\n', 2);
    end
end

