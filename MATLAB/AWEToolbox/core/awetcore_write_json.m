function awetcore_write_json(fd, fn, data)
    if ~exist(fd, 'dir')
        awetcore_log(['Folder ''', fd, ''' does not exist, creating it ...'], 2);
        mkdir(fd)
    end

    json = savejson('', data);
    f = fopen([fd, fn],'w');
    fprintf(f, json);
    fclose(f);
end