% This script do a few additional preprocessing steps on top of HCP
% minimal preprocessing pipeline (Glasser et al 2013)
% 1. Spatial smoothing
% 2. Wishart filtering 
% Replace 'data.nii' to apply on your data

clear
close all
addpath masks
addpath functions/wishart

% Parameters
dataFile='data/data.nii'; % preprocessed fMRI data
FWHM=6; % Gaussian smoothness in mm
voxelsize=2;  % Voxel size in mm
strfound=strfind(dataFile,'.nii');
outFile=[dataFile(1:strfound),'mat']; 

% Load data
insFile='subcortex_mask.nii'; % Subcortex mask 
[~,ins_msk]=read(insFile); ind_ins=find(ins_msk);
gmFile='GMmask.nii'; % Gray matter mask
[~,gm_msk]=read(gmFile); ind_msk=find(gm_msk);

ind_ind_ins=zeros(1,length(ind_ins)); % index of subcortex into gray matter
for i=1:length(ind_ins)
    ind_ind_ins(i)=find(ind_ins(i)==ind_msk);
end

fprintf(['Read file ',dataFile,'\n']);
[~,data]=read(dataFile);

% 1. Spatial smoothing
fprintf('1. Smooth data with FWHM=%dmm\n',FWHM);
T=size(data,4); % Number of time points
x=zeros(T,length(ind_msk));
frst=0;
for i=1:T
    data(:,:,:,i)=imgaussfilt3(data(:,:,:,i),FWHM/voxelsize/2.355);
    tmp=data(:,:,:,i);
    x(i,:)=tmp(ind_msk);
    show_progress(i,T,frst);frst=1;
end
clear data

% 2. Perform Wishart filter. Glasser et al. 2016
fprintf('2. Wishart filtering\n')
DEMDT=1; %Use 1 if demeaning and detrending (e.g. a timeseries) or -1 if not doing this (e.g. a PCA series)
VN=1; %Initial variance normalization dimensionality
Iterate=2; %Iterate to convergence of dim estimate and variance normalization
NDist=1; %Number of Wishart Filters to apply (for most single subject CIFTI grayordinates data 2 works well)

Out=icaDim(x',DEMDT,VN,Iterate,NDist);

x=Out.data';

% Demean and std
x=detrend(x,'constant'); x=x./repmat(std(x),T,1); %remove mean and make std=1
clear i j Out

% Save data
save(outFile,'x');
