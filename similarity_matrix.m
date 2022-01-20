% Script to compute the similarity matrice
% from the preprocessed fMRI scan

clear
close all
addpath masks
addpath functions
addpath functions/gradientography
addpath functions/spm12

% Parameters
dataFile = 'data/data.mat'; % preprocessed fMRI data
TR=0.82; % in seconds
naiveFile='data/task_naive_job.txt'; % naive task file
continuingFile='data/task_continuing_job.txt'; % continuing task file
insFile='subcortex_mask_part1.nii'; % Ventral subcortex mask 
gmFile='GMmask.nii'; % Gray matter mask

% Load data
fprintf(['Loading ',dataFile,'\n'])
load(dataFile, 'x');
T=size(x,1); 

% Compute PPI
regressor_naive=task_regressor(TR,T,naiveFile,0);
regressor_naive=regressor_naive-mean(regressor_naive);

regressor_continuing=task_regressor(TR,T,continuingFile,0);
regressor_continuing=regressor_continuing-mean(regressor_continuing);

PSY=[regressor_naive, regressor_continuing];
fprintf('Compute PPI \n')
ppi=compute_ppi(x,insFile,gmFile,PSY);

% Compute similarity matrice
fprintf('Compute similarity matrix \n')
s = zeros(2, size(ppi,2),size(ppi,2));
for i = 1:2
    zpc=squeeze(ppi(i+1,:,:));
    zpc=zpc(:,all(~isnan(zpc)));
    si=eta_squared(zpc); 
    s(i,:,:)=single(si);
end
s=single(s);
clear zpc

fprintf('Save similarity matrix \n')
save('data/s.mat', 's');
