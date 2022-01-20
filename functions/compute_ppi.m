function Tppi = compute_ppi (x,insFile,gmFile,PSY)
% This script computes psychophysiological interactions (ppi)
% between each pair of subcortical voxels

% INPUT:
% x: fMRI time series, dimension time x number of all gray matter voxels
% Concatenated fMRI signals of all gray matter voxels

% insFile: binary subcortex mask in NIFTI format (*.nii)
% gmFile: binary gray matter mask in NIFTI format (*.nii)

% OUTPUT:
% Tppi: matrix containing psychophysiological interactions (GLM T-stat)

PSY = PSY-min(PSY);
PPIPSY = PSY-mean(PSY);

tic;
%subcortex mask
[~,ins_msk]=read(insFile); ind_ins=find(ins_msk);

%Gray matter mask
[~,gm_msk]=read(gmFile); ind_msk=find(gm_msk);

% index of subcortex into gray matter
ind_ind_ins=zeros(1,length(ind_ins));
for i=1:length(ind_ins)
    ind_ind_ins(i)=find(ind_ins(i)==ind_msk);
end

T=size(x,1); % Number of time points
TR=0.82;

% Demean
x=detrend(x,'constant'); x=x./repmat(std(x),T,1); %remove mean and make std=1

% Subcortex time series
x_ins=x(:,ind_ind_ins);

fprintf('Computing functional connectivity for ROI...\n');

% turn GLM warning off
id =  'stats:glmfit:IterationLimit';
warning('off',id)

if ~any((isnan(x(:)))) % Make sure that all voxels contain no nan
    % define GLM targets (PCA components)
    fprintf('PCA for gray matter time series...\n');
    [U,S,~]=svd(x,'econ');
    a=U*S;
    a=a(:,1:end-1);
    a=detrend(a,'constant');a=a./repmat(std(a),T,1); %remove mean and make std=1
    
    % define GLM task regressors (PSYHRF)
    hrf = spm_hrf(TR);
    k = 1:T;
    PSYHRF = zeros(size(PSY));
    for i = 1:size(PSY,2)
        psyhrf = conv(PSY(:,i),hrf);
        PSYHRF(:,i)=psyhrf(k);
    end
    % create structure for spm_PEB (P)
    xb  = spm_dctmtx(T + 128,T);
    Hxb = zeros(T,T);
    for i = 1:T
        Hx       = conv(xb(:,i),hrf);
        Hxb(:,i) = Hx(k + 128);
    end
    xb = xb(129:end,:);
        % High-pass filter (spm_fmri_concatenate.m)
    s = cumsum([0 T]);
    clear K
    for i=1:numel(T)
        K(i) = struct('HParam', 128,...
                      'row',    s(i) + (1:T),...
                      'RT',     TR);
    end
    K = spm_filter(K);
        % Des Mtx (spm_est_non_sphericity.m)
    xKXs      = spm_sp('Set',spm_filter(K,ones(T,1)));
    xKXs.X    = full(xKXs.X);
    X0 = xKXs.X;
    X0 = [X0 K(1).X0];
    M  = size(X0,2);
    Q = speye(T,T)*T/trace(Hxb'*Hxb);
    Q = blkdiag(Q, speye(M,M)*1e6  );
    clear P
    P{1}.X = [Hxb X0];         % Design matrix for lowest level
    P{1}.C = speye(T,T)/4;      % i.i.d assumptions
    P{2}.X = sparse(T + M,1);   % Design matrix for parameters (0's)
    P{2}.C = Q;
    
    Tppi = zeros(size(PSY,2)*2+2, size(x_ins,2), size(a,2));
    for vj=1:size(x_ins,2)
        fprintf(['PPI (',int2str(vj),'/',int2str(size(x_ins,2)),')\n']);

        % define GLM seed regressor (Y)
        Y=double(x_ins(:,vj));
        
        % define GLM PPI regressor
        C  = spm_PEB(Y,P); % deconvolve H(N)
        xn = xb*C{2}.E(1:T);
        xn = spm_detrend(xn);
        PPI=zeros(size(PSY));
        for i = 1:size(PSY,2)
            PSYxn = PPIPSY(:,i).*xn; % interaction term N*T
            ppi = spm_detrend(conv(PSYxn,hrf)); % reconvolve H(N*T)
            PPI(:,i) = ppi(k);
        end
        
        % GLM
        regressors = [PPI PSYHRF Y];
        for vi=1:size(a,2)
            [b, dev, stats] = glmfit(regressors,a(:, vi));
            %b = glmfit(regressors,a(:, vi));
            for i = 1:size(b,1)
                %Tppi(i, vj,vi)=b(i);
                Tppi(i,vj,vi)=stats.t(i);
            end
        end
    end

else
    fprintf('Error: NAN presented,check your mask\n')
end
toc;



