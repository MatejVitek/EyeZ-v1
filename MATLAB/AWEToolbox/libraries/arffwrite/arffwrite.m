function arffwrite(fname,data,relation)
% fname This is file name without extension
%data is m x n where m are the instances and n-1 are the features. The last
%column is the class as integer

% File modified by Ziga Emersic

sss=size(data,2);
filename1=strcat(fname,'.arff');
out1 = fopen (filename1, 'w+');
aa1=strcat('@relation',{' '},relation);
fprintf (out1, '%s\n', char(aa1));
for jj=1:sss
fprintf (out1, '@attribute attr%s numeric\n',num2str(jj));
end
n_classes=max(unique(data(:,end)));

%fprintf (out1, '%s\n\n',char(txt1));
fprintf (out1,'@data\n');

fclose(out1);

dlmwrite (filename1, data, '-append' );