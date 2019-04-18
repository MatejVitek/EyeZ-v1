function D = awetcore_evaluate_by_distance(A, B)

% chisq = @(a,b) sum( (a-b)^2 / (a+b) ) / 2
% chi2stat = sum((observed-expected).^2 ./ expected)

% in the form of [hist1; hist2; hist3 ...]

A = [1 2; 3 4; 5 6];
B = [1.2 3; 1 4; 5 1];

D = pdist2(A, B, @awetcore_dist_chi);
%disp(D);

%d2 = pdist2([1 2], [1.2 3], @awetcore_dist_chi);
%disp(d2);
