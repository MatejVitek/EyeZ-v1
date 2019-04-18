function awetcore_log(disp_arr, level)
    global awet;
	
    if level <= awet.log_level
        if ~isequal(disp_arr, 'logo')
            fprintf([repmat('\t',1,level), disp_arr, '\n']);
        else
            fprintf([
                  '    ___ _       ________   __              ____              \n' ...
,'   /   | |     / / ____/  / /_____  ____  / / /_  ____  _  __\n' ...
,'  / /| | | /| / / __/    / __/ __ \\/ __ \\/ / __ \\/ __ \\| |/_/\n' ...
,' / ___ | |/ |/ / /___   / /_/ /_/ / /_/ / / /_/ / /_/ />  <  \n' ...
,'/_/  |_|__/|__/_____/   \\__/\\____/\\____/_/_.___/\\____/_/|_|  \n' ...
                ,'                                                        v' ...
                ,awet.VERSION, '\n\n' ...
            ]);
%             fprintf([
%                  '  ______     ___                         _              _ _\n' ...          
%                 ,' / ___\\ \\   / / |       ___  __ _ _ __  | |_ ___   ___ | | |__   _____  __\n' ...
%                 ,'| |    \\ \\ / /| |      / _ \\/ _` | ''__| | __/ _ \\ / _ \\| | ''_ \\ / _ \\ \\/ /\n' ...
%                 ,'| |___  \\ V / | |___  |  __/ (_| | |    | || (_) | (_) | | |_) | (_) >  <\n' ...
%                 ,' \\____|  \\_/  |_____|  \\___|\\__,_|_|     \\__\\___/ \\___/|_|_.__/ \\___/_/\\_\\\n' ...
%                 ,'                                                                      v' ...
%                 ,awet.VERSION, '\n\n' ...
%             ]);
        end
    end
end