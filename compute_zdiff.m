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
        compute_zdiff([resultFolder,'/cohorts/',char(grp),'/tasks'], Vn, nperm);
    end
end

%% Compute z-diff: BACKGROUND - TASKS
for Vn=[2]
    for grp={'cc_c_rs','cc_n_rs','hc_c_rs','hc_n_rs'}
        for task={'naive', 'continuing', 'resting_state'}
            if isfolder([resultFolder,'/cohorts/',char(grp),'/tasks/',char(task)])
                grp=char(grp);
                source=[resultFolder,'/tasks/',char(task),'/cohorts/',grp(1:2),'/Vn',int2str(Vn),'_magnitude.nii'];
                destination=[resultFolder,'/cohorts/',char(grp),'/tasks/',char(task)];
                copyfile(source, destination);
            end
        end
        compute_zdiff([resultFolder,'/cohorts/',char(grp),'/tasks'], Vn, nperm);
    end
end


   

