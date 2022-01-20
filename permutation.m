% This script do a permutation test

clear
close all
addpath masks
addpath functions/gradientography
addpath functions/spm12

% Parameters
perm=1; % random seed
rng(perm); % shuffle random seed for datasample
insFile = 'subcortex_mask_part1.nii';
roiFile = 'subcortex_mask_part1.nii';
%similarityFolder = 'data/similarity';
similarityFolder='/home/leonie/Documents/result/PISA/Result/gradientography_TSTAT_with_confounds/similarity';
Vn=2;

% fixed task, permuted cohorts
%%resultFolder='result/tasks/continuing/cohorts';
%resultFolder='result/tasks/naive/cohorts';
%permArray={'hc', 'cc'};

% fixed cohort, permuted tasks
%resultFolder='result/cohorts/hc/tasks';
resultFolder='result/cohorts/cc/tasks';
permArray={'naive', 'continuing'}; 

% define subj and task array
if any(contains(permArray, 'naive'))
    folderLim = strfind(resultFolder,'/');
    subjArray1Perm = readcell([resultFolder(1:folderLim(end)),'subjects.txt']);
    subjArray2Perm = subjArray1Perm;
%     taskArray1Perm = datasample(1:2, length(subjArray1Perm));
    taskArray = repmat(1:2, 1, length(subjArray1Perm));
    permid=randperm(size(taskArray, 2));
    taskArray1Perm = taskArray(permid);
    taskArray2Perm = taskArray1Perm*-1+3;
else
    subjArray1 = readcell([resultFolder,'/',char(permArray(1)),'/subjects.txt']);
    subjArray2 = readcell([resultFolder,'/',char(permArray(2)),'/subjects.txt']);
    subjArray = [subjArray1 subjArray2];
    permid = randperm(size(subjArray,2));
    subjArray1Perm =  subjArray(permid(1:size(subjArray1,2)));
    subjArray2Perm = subjArray(permid(size(subjArray1,2)+1:end));
    if contains(resultFolder, 'naive')
        taskArray1Perm = zeros(length(subjArray1)) + 1;
        taskArray2Perm = zeros(length(subjArray2)) + 1;
    elseif contains(resultFolder, 'continuing')
        taskArray1Perm = zeros(length(subjArray1)) + 2;
        taskArray2Perm = zeros(length(subjArray2)) + 2;
    end
end

% run permutation
fprintf(['\nPERMUTATION ',int2str(perm),'\n'])
for folder=1:2

    if folder==1
        subjArray = subjArray1Perm;
        taskArray = taskArray1Perm;
    else 
        subjArray = subjArray2Perm;
        taskArray = taskArray2Perm;
    end

    permFolder = [resultFolder, '/', char(permArray(folder)), '/permutation/'];
    mkdir(permFolder);

    % 1.Compute subcortex-to-subcortex similarity matrix 
    fprintf('\n1.Computing similarity matrices\n')
    savg = zeros(4510,4510); % list of similarity matrices
    ndata=0;
    for i = 1:length(subjArray)
        dataFile = ['similarity_fmri_fix_',char(subjArray(i)),'.mat']; %%%%
        fprintf(['Include ',dataFile,'\n'])
        similarityFile=[similarityFolder,'/',dataFile];
        load(similarityFile, 's');
        savg=savg+squeeze(s(taskArray(i),:,:));
        ndata=ndata+1;
    end

    fprintf('Group-averaged similarity matrix\n')
    savg=savg/ndata;
    
    Prefix=[permFolder, '/', num2str(perm),'_'];
    Subject=[Prefix,'Vn',num2str(Vn),'_'];

    % 2.Map functional connectivity gradient
    fprintf('\n2.Map functional connectivity gradient\n')
    [~,ins_msk]=read(insFile);
    ind_ins_org=find(ins_msk); % The original subcortical mask
    Streamlines=0; % Streamlines:1-> write out vector file in /tmp; 0->do not write out vector file; 
    Fig=0; % Figures: 0->suppress figures; 1->print figures 
    cont_model(savg,ind_ins_org,roiFile,Fig,Streamlines,Prefix,Vn);
end



