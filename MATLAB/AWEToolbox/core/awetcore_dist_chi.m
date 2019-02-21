%function [ D ] = awetcore_dist_chi(A, B)
%    D = sum(A, 2) - sum(B, 2);
%end

function D = awetcore_dist_chi(A, B)
  
  m = size(B,1); % number of samples of p
  p=size(A,2); % dimension of samples
  
  assert(p == size(B,2)); % equal dimensions
  assert(size(A,1) == 1); % pdist requires XI to be a single sample
  
  D = zeros(m,1); % initialize output array
  
  for i=1:m
    for j=1:p
      m=(A(1,j) + B(i,j)) / 2;
			if m ~= 0 % if m == 0, then xi and xj are both 0 ... this way we avoid the problem with (xj - m)^2 / m = (0 - 0)^2 / 0 = 0 / 0 = ?
				D(i,1) = D(i,1) + ((A(1,j) - m)^2 / m); % B is the model! makes it possible to determine each "likelihood" that A was drawn from each of the models in XJ
			end
    end
  end

