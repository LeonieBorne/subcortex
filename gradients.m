% Script to compute the gradients
% from the similarity matrices

clear
close all
addpath masks
addpath functions/gradientography

% Parameters
similarityFolder='data/similarity';
resultFolder='result';
grFolders={'hc', 'cc'};
grids={{'subject1.mat', 'subject2.mat'},
       {'subject3.mat', 'subject4.mat', 'subject5.mat'}};
%grids={readcell('result/tasks/continuing/cohorts/hc/subjects.txt'),
%       readcell('result/tasks/continuing/cohorts/cc/subjects.txt')};
taskFolders={'naive', 'continuing'};

insFile = 'subcortex_mask_part1.nii';
roiFile = 'subcortex_mask_part1.nii';
Vn=3;

% Compute gradients
for t=1:size(taskFolders,2)
    for g=1:size(grids)
        fprintf(['\nTASK ',char(taskFolders{t}),' COHORT ',char(grFolders{g}),'\n']);
        rpath=[resultFolder,'/tasks/',char(taskFolders{t}),'/cohorts/',char(grFolders{g}),'/'];
        mkdir(rpath);
        incids=grids{g};
        writecell(incids, [rpath,'subjects.txt']);

        % 1.Compute subcortex-to-subcortex similarity matrix 
        fprintf('\n1.Computing group-averaged similarity matrix\n')
        savg = zeros(4510,4510); % list of similarity matrices
        ndata=0;
        for i = 1:length(incids)
            subj = char(incids{i});
            %subj = ['similarity_fmri_fix_' char(incids{i}) '.mat'];
            fprintf(['Include ',subj,'\n'])
            similarityFile=[similarityFolder,'/',subj];
            load(similarityFile, 's');
            savg=savg+squeeze(s(t,:,:));
            ndata=ndata+1;
        end
        savg=savg/ndata;
        save([rpath,'savg.mat'],'savg');   
%        load([rpath,'savg.mat'],'savg');

        % 2.Map functional connectivity gradient
        Prefix=rpath;
        Subject=[Prefix,'Vn',num2str(Vn),'_'];
        fprintf('\n2.Map functional connectivity gradient\n')
        [~,ins_msk]=read(insFile);
        ind_ins_org=find(ins_msk); % The original subcortical mask
        Streamlines=1; % Streamlines:1-> write out vector file; 0->do not write out vector file; 
        Mag=1; % Figures: 0->suppress figures; 1->print figures 
        cont_model(savg,ind_ins_org,roiFile,Mag,Streamlines,Prefix,Vn);

        fprintf(['\nResults saved in ', rpath,'\n']);
    end
end

