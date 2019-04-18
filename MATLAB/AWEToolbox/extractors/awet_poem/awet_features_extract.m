function [features] = awet_features_extract(I, I_annotations)
    %features = awet_features_extract_old(I, I_annotations);
    mapping = getmapping(8,'u2');
	
	% extract ear features for current image
	%
	% Input:
	%    I             = already preprocessed ear image
	%    I_annotations = annotation data for this image
	%
	% Output:
	%    features      = vector of ear features
    
    % POEM practical:
    % 1. calculate AEMIs
    % 2. apply LBP on those for each block (you then get POEM images)
    % 3. calc histograms on divided POEMs and contencate them
    
    
    CELL_SIZE = 7; % sum parameter of gradients - 7 is in paper
    BLOCK_SIZE = 10; % LBP parameter - 10 is in paper
    REGION_SIZE = 8; %  POEM hist calculation - 8 is in paper
    BIN_SIZE = 8; % number of bins in histogram - number not in paper
    
    % 1. EMI calculation
    % TODO: trenutno dela boljse ce ima 64, true. Vrednosti 3, false sta po
    % clanku
    [Imag, IdirRad, uniqs] = emi(I, 3, false);
    
    uniqsLen = size(uniqs, 1);
    
    % 2. m x UEMI
    EMIs = zeros(uniqsLen, size(Imag, 1), size(Imag, 2));    
    for i = 1:uniqsLen
        EMIs(i,:,:) = Imag .* (IdirRad == uniqs(i));        
    end
    
    % 3.,4. AEMI calculation
    % a + d - b - c  
    %AEMIs = zeros(uniqsLen, size(Imag, 1) - CELL_SIZE + 1, size(Imag, 2) - CELL_SIZE + 1);
    
%     POEM_HS = uint8(zeros(uniqsLen, (ceil( (size(Imag, 1) - CELL_SIZE + 1) /REGION_SIZE)^2) * BIN_SIZE));
    %POEM_HS = zeros(uniqsLen, BIN_SIZE);
    POEM_HS = zeros(uniqsLen, 8*472);
    for i = 1:uniqsLen
        In = integralImage(squeeze(EMIs(i,:,:)));
        %%%integralSum = In(r, c) + In(r + h, c + w) - In(r, c + w) - In(r + h, c);
        Ikernel = integralKernel([1 1 CELL_SIZE CELL_SIZE], 1);
        %%%EMIs(i,:,:) = integralFilter(In, Ikernel);
        AEMI = integralFilter(In, Ikernel);
        
        % 5. apply LBP on AEMIs
        LBP = lbpMatlab(AEMI, mapping);
        
%         fun = @(bl) histcounts(bl.data, 59);
%         POEM = blockproc(LBP, [REGION_SIZE REGION_SIZE], fun);
        %POEM = histcounts(LBP, BIN_SIZE);
        POEM = blockproc(double(LBP),[12,12], @(x) hist(x.data(:)',0:58),'BorderSize',[4,4],'TrimBorder',false);
%         whos POEM
        
        POEM_HS(i,:) = POEM(:)';
    end
    features = POEM_HS(:)';
    
    %features = lbpSelf(I);
    
end

function [Imag, IdirRad, uniqs] = emi64(I)
    %figure; imshow(I, []);
    [Imag, Idir] = imgradient(I);
    IdirRad = (Idir + 180) * pi / 180;
    %figure; imshowpair(Imag, IdirRad, 'montage');
    
    
    % This makes 64 values:
    IdirRad = round(IdirRad, 1);
    %uniqs = unique(IdirRad); % this was run only the first time
    uniqs = (0:0.1:6.3)';
end

function [Imag, IdirRad, uniqs] = emi(I, step, signed)
    [Imag, Idir] = imgradient(I);
    if (signed)
        Idir = Idir + 180;
        top = 2*pi;
    else
        Idir = abs(Idir);
        top = pi;
    end
    IdirRad = Idir * pi / 180;
    
    % Discretize to as many values as in step
    IdirRad = quant(IdirRad, (top / (step-1)));
    uniqs = (0:(top / (step-1)):(top))';
end

function features = lbpMatlab(I, mapping)
%     nFiltSize = 8;
%     nFiltRadius = block_size;
    I=padarray(I',2,'replicate','both');
    I=padarray(I',2,'replicate','both');
    features=lbp_SAM(I,2,8,mapping,'i');
end

function features = lbpSelf(I)
    hist = zeros(1, 256);
    for y = 2 : (size(I,1) - 1)
        for x = 2: (size(I,2) - 1)
            code = getBinaryCode(I, x, y);
            if (code > 0)
                hist(code) = hist(code) + 1;
            end
        end
    end
    features = normr(hist);
end

function code = getBinaryCode(I, x, y)
    xs = [-1 0 1 1 1 0 -1 -1];
    ys = [-1 -1 -1 0 1 1 1 0];
    m = [1 2 4 8 16 32 64 128];
    len = size(xs, 2);
    c = I(y, x);
    xsize = size(I, 2);
    ysize = size(I, 1);
    code = 0;
    transitions = 0;
    
    prev = -1;
    for i = 1:len
        posX = mod(x + xs(i), xsize);
        posY = mod(y + ys(i), ysize);
        
        if (posX < 1)
            posX = xsize + posX;
        end
        if (posY < 1)
            posY = ysize + posY;
        end    
        
        el = 0;
        if (I(posY, posX) >= c)
            code = code + m(i);
            el = 1;
        end
        if (prev == 0 && el == 1)
            transitions = transitions + 1;
        end
        prev = el;
    end
    if (transitions > 2)
        code = 0;
    end
end