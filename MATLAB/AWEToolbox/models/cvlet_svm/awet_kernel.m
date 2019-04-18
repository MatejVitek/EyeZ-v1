function G = awet_kernel(U,V)
    % Sigmoid kernel function with slope gamma and intercept c
    %gamma = 1;
    %c = -1;
    %G = tanh(gamma*U*V' + c);
    G = abs(U - V);
    size(G)
    disp(sum(G));
end