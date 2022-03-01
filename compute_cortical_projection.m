% This script compute the cortical projection of the eigenvector 

clear
close all
addpath masks
addpath functions/gradientography
addpath functions/spm12

resultFolder = 'result'; 

% load masks
insFile='subcortex_mask_part1.nii'; % Ventral subcortex mask 
[~,ins_msk]=read(insFile); ind_ins=find(ins_msk);
subFile='subcortex_mask.nii'; % Subcortex mask 
[~,sub_msk]=read(subFile); ind_sub=find(sub_msk);
gmFile='GMmask.nii'; % Gray matter mask
[~,gm_msk]=read(gmFile); ind_gm=find(gm_msk);

hipFile='hippocampus.nii';
[~,hip_msk]=read(hipFile); ind_hip=find(hip_msk);
ind_ind_hip=zeros(1,length(ind_hip)); % index of hippocampus into ventral subcortex
for i=1:length(ind_hip)
    ind_ind_hip(i)=find(ind_hip(i)==ind_ins);
end
        
for grp = {'hc', 'cc'}
    fprintf(['\n **** ',char(grp),' **** \n']);
    mkdir([resultFolder,'/projection/',char(grp)]);
    subjArray = readcell([resultFolder,'/tasks/naive/cohort/',char(grp),'/subjects.txt']);

    %% PPI projection on the cortex 
	% compute projection matrix from ppi
	outFile=[resultFolder,'/projection/',char(grp),'/ppic.mat'];
	if ~isfile(outFile)
	    fprintf('Projecting PPI...\n');
	    ppic = zeros(length(ind_ins), length(ind_gm));
	    nsubj=length(subjArray);
        
	    for subja=subjArray
            subj = char(subja);
            fprintf([subj,'\n']);
            
            % load ppi
            dataFile = ['fmri_fix_' char(subj) '.mat']; %%%
            load([resultFolder,'/ppi/ppi_',dataFile], 'Bppi');
            ppi = squeeze(Bppi(end,:,:));

            % compute pca
            matFolder=[resultFolder,'/mat'];
            load([matFolder,'/',dataFile], 'x');
            T=size(x,1); % Number of time points
            x=detrend(x,'constant'); x=x./repmat(std(x),T,1); %remove mean and make std=1
            [~,~,V]=svd(x,'econ');

            % project ppi on the whole cortex
            ppic = ppic + [ppi, zeros(size(ppi,1),1)]*V';
        end
	    ppic = ppic/nsubj;
        save(outFile, 'ppic', '-v7.3'); %%%
	else
	    fprintf('Loading ppic...\n');
	    load(outFile, 'ppic');
	end

	%% Cortex-to-hip
    outFile=[resultFolder,'/projection/',char(grp),'/ind_max_ppic_',hipFile(1:end-4),'.mat'];
    fprintf('Finding cortex-to-hip...\n');
    % connectivity matrix cortex-to-subcortex
    max_ppic = max(ppic(ind_ind_hip,:)); % select hip only
    ind_max_ppic = zeros(1,length(max_ppic));
    for i=1:length(max_ppic)
        idx=find(ppic(:,i) == max_ppic(i));
        if length(idx)==1
            ind_max_ppic(1,i) = idx;
        else
            ind_max_ppic(1,i) = idx(1);
        end
    end
    save(outFile, 'ind_max_ppic');

    %% Gradients and magnitude projection
    for task = {'naive', 'continuing'}
        Prefix = [resultFolder,'/projection/',char(grp),'/',char(task),'/'];
        for Vn=2:3
            eigvecFile = [Prefix,'Vn',int2str(Vn),'_eigenvector.nii'];
            if ~isfile(eigvecFile)
                % similarity matrice
                fprintf('\n1.Computing similarity matrices\n')
                savg = zeros(4510,4510);
                ndata=0;
                for subj = subjArray
                    dataFile = ['fmri_fix_',char(subj),'.mat']; %%%
                    id=dataFile(10:12);
                    fprintf(['Include ',id,'\n'])
                    similarityFile=[resultFolder,'/similarity/similarity_',dataFile];
                    load(similarityFile, 's');
                    if strcmp(task, 'naive')
                        savg=savg+squeeze(s(1,:,:));
                    else
                        savg=savg+squeeze(s(2,:,:));
                    end
                    ndata=ndata+1;
                end

                fprintf('Group-averaged similarity matrix\n')
                savg=savg/ndata;

                % compute gradients
                fprintf('\n2.Map functional connectivity gradient\n')
                [~,ins_msk]=read(insFile);
                ind_ins_org=find(ins_msk); % The original subcortical mask
                Streamlines=0; % Streamlines:1-> write out vector file in /tmp; 0->do not write out vector file; 
                Mag=0; % Figures: 0->suppress figures; 1->print figures 

                mkdir(Prefix);
                cont_model(savg,ind_ins_org,insFile,Mag,Streamlines,Prefix,Vn);
            end
            
            [~,eigvec] = read(eigvecFile);

            % project eigenmap
            fprintf(['Project eigenmap for ',char(task),' task, gradient ',int2str(Vn-1),'...\n'])
            ind_eig=find(eigvec ~= 0);
            eig=eigvec(ind_eig);
            eig_proj = eig(ind_max_ppic);
            [xx,yy,zz]=ind2sub(size(gm_msk),ind_gm);
            eigvec_proj=zeros(size(gm_msk));
            for i=1:length(ind_gm)
                if ~ismember(ind_gm(i), ind_sub) % set subcortex to 0
                    eigvec_proj(xx(i),yy(i),zz(i))=eig_proj(i);
                end
            end
            mat2nii(eigvec_proj, [Prefix,'Vn',int2str(Vn),'_eigenvector_projection_',hipFile],size(eigvec_proj),32,gmFile);
            
        end
    end
end
