% This script do compute the z-scored magnitude
% difference between groups and tasks

clear
close all
addpath masks
addpath functions/gradientography
addpath functions/spm12
addpath functions

nperm=1000;
resultFolder='result';

%% Compute z-diff: GR1 - GR2
folderArray = 'cohorts';
Vn=2;

for task = {'naive', 'continuing'}
     compute_zdiff([resultFolder,'/tasks/',char(task),'/cohorts'], Vn, nperm);
end

%% Compute z-diff: NAIVE - CONTINUING
for Vn=[2 3]
    for g=1:2
        if g==1
            grp='cc';
        else
            grp='hc';
        end
        compute_zdiff(['cohort/',char(grp),'/tasks'], Vn, nperm);
    end
end


   

