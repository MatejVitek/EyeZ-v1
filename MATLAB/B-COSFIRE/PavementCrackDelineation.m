function [ ] = PavementCrackDelineation( )

    % NOTE: It requires a compiled mex-file of the fast implementation 
    % of the max-blurring function.
    if ~exist('./COSFIRE/dilate')
        BeforeUsing();
    end
    
    % temporary
    dataset = 2;
    CRACK_IVC = 1;
    CRACK_PV14 = 2;

    if dataset == CRACK_IVC
        error('Dataset not yet available.')
        return;
    elseif dataset == CRACK_PV14
        dname = 'CrackPV14';
        imagesdir = 'cracks14';        
        gtdir = 'cracks14_gt';
        prefix_gt = '';
    end


    % Binarization thresholds
    thresholds = 0.01:0.01:0.99;
    nthresholds = numel(thresholds);

    %% Symmetric filter params and configuration
    x = 101; y = 101; % center
    line1(:, :) = zeros(201);
    line1(:, x) = 1; %prototype line

    % Parameters determined in the paper 
    % N.Strisciuglio, G. Azzopardi, N.Petkov, "Detection of curved lines 
    % with B-COSFIRE filters: A case study on crack delineation", CAIP 2017
    sigma = 3.3;
    len = 14;
    sigma0 = 2;
    alpha = 1;
    
    % Symmetric filter params
    symmfilter = cell(1);
    symm_params = SystemConfig;
    % COSFIRE params
    symm_params.inputfilter.DoG.sigmalist = sigma;
    symm_params.COSFIRE.rholist = 0:2:len;
    symm_params.COSFIRE.sigma0 = sigma0 / 6;
    symm_params.COSFIRE.alpha = alpha / 6;
    % Orientations
    numoriens = 12;
    symm_params.invariance.rotation.psilist = 0:pi/numoriens:pi-pi/numoriens;
    % Configuration
    symmfilter{1} = configureCOSFIRE(line1, round([y x]), symm_params);

    % Prepare the filter set
    filterset(1) = symmfilter;

    %% APplication of B-COSFIRE for crack delineation
    files = rdir(['./data/' dname '/' imagesdir  '/*.bmp']);
    nfiles = size(files, 1);

    % Initialize result matrix
    nmetrics = 3;
    RESULTS = zeros(nfiles + 1, nmetrics, nthresholds);
    
    for n = 1:nfiles
        fprintf('Processing image %d of %d. ', n, nfiles);
        % Read image
        imageInput = double(imread(files(n).name)) ./ 255;
        % Read groud truth
        [p, name, ext] = fileparts(files(n).name);
        gt = double(imread(['./data/' dname '/' gtdir '/' prefix_gt name '.bmp'])) ./ 255;

        imageInput = imcomplement(imageInput);

        % Pad input image to avoid border effects
        NP = 50; imageInput = padarray(imageInput, [NP NP], 'replicate');

        % Filter response
        inhibFactor = 0;
        tuple = computeTuples(imageInput, filterset);
        [response, rotations] = applyCOSFIRE_inhib(imageInput, filterset, inhibFactor, tuple);
        response = response{1};
        response = response(NP+1:end-NP, NP+1:end-NP);
        % Cropping out the central part (unpadding)
        rotations_final = zeros(size(response, 1), size(response, 2), size(rotations, 3));
        for j = 1:size(rotations, 3)
            rotations_final(:,:,j) = rotations(NP+1:end-NP, NP+1:end-NP, j);
        end

        % Evaluation
        fprintf(' Result evaluation...\n');
        for j = 1:nthresholds
            % Thinning and Histeresis thresholding (using different
            % thresholds). The threshold of the CAIP17 paper is th=49. Here
            % we compute the performance for differen thresholds anyway to
            % build and show the ROC curve.
            binImg = binarize(rotations_final, thresholds(j));
            
            binImg2 = bwmorph(binImg, 'close');
            binImg2 = bwmorph(binImg2,'skel',Inf);
            % Compute the result metrics for a tolerance of d = 2 (as in
            % the paper)
            [cpt2, crt2, F2] = evaluate(binImg2, gt, 2);
            %[cpt3, crt3, F3] = evaluate(binImg2, gt, 3);

            RESULTS(n, :, j) = [cpt2, crt2, F2];
        end

    end
    
    % Average Results
    avg_results = reshape(mean(RESULTS(1:nfiles, :, :)), nmetrics, nthresholds)';
    [M, idx] = max(avg_results(:,3));
    
    fprintf('\nResults of the CAIP17 paper\n');
    fprintf('Pr: %.3f, Re: %.3f, F: %.3f\n', avg_results(idx, 1), avg_results(idx, 2), avg_results(idx, 3));
    
    PrintROCcurve(avg_results);
end

function [cpt, crt, F] = evaluate(binImg, gt, d)

    A = zeros(d*2+1, d*2+1); A(d+1,d+1) = 1; B = bwdist(A) <= d;
    
    [m, n] = size(binImg);
    %binImg = bwmorph(binImg,'skel',Inf);
    gt = padarray(gt, [d d], 0);
    %binImg = padarray(binImg, [d d], 0);
    Lr = 0;

    bad = zeros(size(binImg));
    for x = 1:m
        for y = 1:n
            %if gt(x, y) == 1 
            if binImg(x,y) == 1
                %patch = binImg(x:x+2*d, y:y+2*d); %
                patch = gt(x+d-d:x+d+d, y-d+d:y+d+d);
                s = sum(patch(:) .* B(:));
                if s > 0
                    Lr = Lr + 1;
                else
                    bad(x,y) = 1;
                end
            end
        end
    end
    Lgt = sum(gt(:));
    Ln = sum(binImg(:));
    
    cpt = min(1, Lr / Lgt);
    crt = min(1, Lr / Ln);
    F = 2 * cpt * crt / (cpt + crt);
end

function [binarymap] = binarize(rotoutput1, highthresh)
    %%%%%%%%%%%%%%%%% BEGIN BINARIZATION %%%%%%%%%%%%%%%%%%
    % compute thinning
    orienslist = 0:pi/12:pi-pi/12;
    [viewResult, oriensMatrix] = calc_viewimage(rotoutput1,1:numel(orienslist), orienslist);
    thinning = calc_thinning(viewResult, oriensMatrix, 1);
    %figure; imagesc(thinning);
    % 
    % % Choose high threshold of hysteresis thresholding
    % if nargin == 4
    %     bins = 64;p = 0.05; %Keep the strongest 10% of the pixels in the resulting thinned image
    %     f = find(thinning > 0);
    %     counts = imhist(thinning(f),bins);
    %     highthresh = find(cumsum(counts) > (1-p)*length(f),1,'first') / bins;
    % end
    % 
    binarymap = calc_hysteresis(thinning, 1, 0.5*highthresh*max(thinning(:)), highthresh*max(thinning(:)));
    %figure;imagesc(binarymap);colormap gray; axis image;
    % show binarized image
    % figure;
    % subplot(1,2,1);imagesc(img);axis off;axis image;colormap(gray);
    % subplot(1,2,2);imagesc(imcomplement(binarymap));axis off;axis image;colormap(gray);
    %%%%%%%%%%%%%%%%% END BINARIZATION %%%%%%%%%%%%%%%%%%%%
end

function [] = PrintROCcurve(avg_results)
    pr = avg_results(:,1);
    re = avg_results(:,2);

    figure;
    linewidth = 3;
    plot(re, pr, 'linewidth',linewidth,'color',[0.25 0.25 0.25],'markersize',10);
    set(gca,'YGrid','off');
    set(gca,'XGrid','off');
    set(gca,'XTick',0:.1:1)
    set(gca,'XTickLabel',0:.1:1)
    axis square;

    % Plot other methods results
    hold on;
    % 1: Zou14 - data set
    % 2: CrackTree
    % 3: FoSA
    Pr = [0.872 0.821 0.845;
    0.842 0.625 0.733;
    0.846 0.885 0.897; 
    0.793 0.753 0.756; 
    0.949 0.845 0.860;
    0.671 0.780 0.836; 
    0.960 0.698 0.716; 
    0.846 0.696 0.749; 
    0.767 0.722 0.779; 
    0.833 0.927 0.811;
    0.833 0.839 0.792; 
    0.997 0.847 0.868; 
    0.499 0.775 0.696; 
    0.848 0.948 0.925]; 

    Re = [0.965 0.691 0.628 ;
    0.904 0.605 0.568 ;
    0.905 0.713 0.612 ;
    0.903 0.776 0.691 ;
    0.939 0.600 0.577;
    0.843 0.649 0.647 ;
    0.915 0.605 0.552 ;
    0.929 0.668 0.654 ;
    0.996 0.669 0.636 ;
    0.961 0.860 0.805;
    0.993 0.967 0.937 ;
    0.823 0.923 0.805 ;
    0.890 0.706 0.663 ;
    0.988 0.985 0.880];

    F = [0.916 0.751 0.721 ;
    0.872 0.614 0.640 ;
    0.874 0.790 0.728 ;
    0.845 0.764 0.722 ;
    0.944 0.700 0.691;
    0.747 0.708 0.729 ;
    0.937 0.648 0.623 ;
    0.886 0.682 0.698 ;
    0.867 0.695 0.700 ;
    0.892 0.892 0.808;
    0.906 0.898 0.858 ;
    0.893 0.883 0.835 ;
    0.639 0.739 0.679 ;
    0.913 0.966 0.901];

    avgPr = mean(Pr);
    avgRe = mean(Re);
    avgF = mean(F);
    linewidth = 2;
    plot(avgRe(1), avgPr(1), 'o', 'linewidth',linewidth,'color',[0.25 0.25 0.25],'markersize',10);
    plot(avgRe(2), avgPr(2), 's', 'linewidth',linewidth,'color',[0.25 0.25 0.25],'markersize',10);
    plot(avgRe(3), avgPr(3), 'd', 'linewidth',linewidth,'color',[0.25 0.25 0.25],'markersize',10);
    xlabel('Recall');
    ylabel('Precision');
    legend({'COSFIRE', 'Zou14', 'CrackTree', 'FoSA'}, 'Location', 'southwest');
    title('ROC curve');
    
    fprintf('\nResults of Zou et al.\n');
    fprintf('Pr: %.3f, Re: %.3f, F: %.3f\n', avgPr(1), avgRe(1), avgF(1));
    fprintf('\nResults of CrackTree.\n');
    fprintf('Pr: %.3f, Re: %.3f, F: %.3f\n', avgPr(2), avgRe(2), avgF(2));
    fprintf('\nResults of FoSA.\n');
    fprintf('Pr: %.3f, Re: %.3f, F: %.3f\n', avgPr(3), avgRe(3), avgF(3));
    
end