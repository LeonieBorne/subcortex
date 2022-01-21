% This script do compute the variance explained

clear
close all
addpath masks
addpath functions/gradientography
addpath functions/spm12

resultFolder = 'result'; 

%% Set parameters
for g=1:2
    for t=1:2
        
        if t==1
            task='naive';
            fprintf('\n **** NAIVE **** \n');
        else 
            task='continuing';
            fprintf('\n **** CONTINUING **** \n');
        end
        if g==1
            grp='hc';
            fprintf(['\n **** HC **** \n']);
        else
            grp='cc';
            fprintf(['\n **** CC **** \n']);
        end
        
        roiFile='subcortex_mask_part1.nii';
        [~,name]=fileparts(roiFile);

        insFile='subcortex_mask_part1.nii'; % Subcortex mask 
        gmFile='GMmask.nii'; % Gray matter mask
        
        load([resultFolder, '/tasks/', task, '/cohorts/', grp, '/savg.mat'],'savg');    

        [~,ins_msk]=read(insFile);
        ind_ins_org=find(ins_msk); % The original subcortical mask
        Streamlines=0; % Streamlines:1-> write out vector file in /tmp; 0->do not write out vector file; 
        Mag=1; % Figures: 0->suppress figures; 1->print figures 

        %% cont_model(savg,ind_ins_org,roiFile,Mag,Streamlines,Prefix,Vn);

        % Compute gradient for roi
        [~,roi_msk]=read(roiFile);
        ind_roi=find(~~roi_msk);
        N=size(roi_msk);

        % index of roi into the whole subcortex
        ind_ind_trim=zeros(1,length(ind_roi));
        for i=1:length(ind_roi)
            ind_ind_trim(i)=(find(ind_roi(i)==ind_ins_org));
        end

        % Similarity matrix for voxels in roi
        s=savg(ind_ind_trim,ind_ind_trim);

        ind_ins=ind_roi;
        ins_msk=zeros(size(roi_msk));
        ins_msk(ind_ins)=1;

% hf=figure;
% imagesc(s);
% colormap(flipud(bone))
% set(gca,'xtick',[])
% set(gca,'ytick',[])
% saveas(hf,['/tmp/savg.png']);

        %% img_pca=connectopic_laplacian(s,ind_ins,N,Vn);

        %Global thresholding. Haak et al 2017
        w=squareform(pdist(s));  %similarity to distance mapping

        fprintf('Thresholding to minimum density needed for graph to remain connected\n');
        ind_upper=find(triu(ones(length(w),length(w)),1));
        [~,ind_srt]=sort(w(ind_upper));
        w_thresh=zeros(length(w),length(w));
        dns=linspace(0.001,1,1000);
        for i=1:length(dns)
            ttl=ceil(length(ind_upper)*dns(i));
            w_thresh(ind_upper(ind_srt(1:ttl)))=s(ind_upper(ind_srt(1:ttl)));
            [~,comp_sizes]=get_components(~~w_thresh+~~w_thresh');
            if length(comp_sizes)==1
                break
            end
        end

        fprintf('Density=%0.2f%%\n',100*(length(find(~~w_thresh))/length(ind_upper)));
        dns=dns(i);
        w_thresh=w_thresh+w_thresh';

% hf=figure;
% imagesc(w_thresh);
% colormap(flipud(bone))
% set(gca,'xtick',[])
% set(gca,'ytick',[])
% saveas(hf,['/tmp/w_thresh.png']);

        fprintf('Computing Laplacian\n');
        L=diag(sum(w_thresh))-w_thresh;

        fprintf('Finding eigenvectors\n');
        [eigenv,eigend]=eig(L);d=diag(eigend);

% hf=figure;
% imagesc(eigenv(:,1:3));
% colormap(flipud(bone))
% set(gca,'xtick',[])
% set(gca,'ytick',[])
% c=colorbar
% c.Ticks=[] 
% saveas(hf,['/tmp/gradients.png']);

        % Variance explained
        per=1./d(2:end);
        per=per/sum(per)*100;
        fprintf('Variance explained Gradient I %0.2f%%\n',per(1));
        fprintf('Variance explained Gradient II %0.2f%%\n',per(2));
        fprintf('Variance explained Gradient III %0.2f%%\n',per(3));
        writematrix(per,[resultFolder, '/tasks/', task, '/cohorts/', grp, '/variance_explained.csv']) 
    end
end
