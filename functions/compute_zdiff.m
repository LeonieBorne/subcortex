function [] = compute_zdiff(folder, Vn, nperm)

    % find folders
    files = dir(folder);
    grpArray = {};
    for k = 1 : length(files)
        if files(k).isdir
            if ~startsWith(files(k).name, '.')
                grpArray{end+1} = files(k).name;
            end
        end
    end
    if length(grpArray) ~= 2
        fprintf('\nERROR: too many groups detected\n')
    end
    grpArray = sort(grpArray);

    % load ref magnitude nii
    data1_file=[char(folder),'/',char(grpArray{1}),'/Vn',int2str(Vn),'_magnitude.nii'];
    data2_file=[char(folder),'/',char(grpArray{2}),'/Vn',int2str(Vn),'_magnitude.nii'];
    [~, data1_ref] = read(data1_file);
    [~, data2_ref] = read(data2_file);
    fprintf([pwd,'/',data1_file,'\n']);
    fprintf([pwd,'/',data2_file,'\n']);

    data_ref = data1_ref-data2_ref;
    data_ref_inv = data2_ref-data1_ref;
    
    diff_file = [char(folder),'/Vn',int2str(Vn),'_',char(grpArray{1}),'-',char(grpArray{2}),'.nii'];
    fprintf([pwd,'/',diff_file,'\n']);
    mat2nii(squeeze(data_ref),diff_file);
    fprintf('\n');
    
    % load permutation magnitude nii
    data_all = zeros(size(data_ref,1), size(data_ref,2), size(data_ref,3), nperm);
    data_all_inv = zeros(size(data_ref,1), size(data_ref,2), size(data_ref,3), nperm);
    for perm=1:nperm
        [~, data1] = read([char(folder),'/',char(grpArray{1}),'/permutation/',int2str(perm),'_Vn',int2str(Vn),'_magnitude.nii']);
        [~, data2] = read([char(folder),'/',char(grpArray{2}),'/permutation/',int2str(perm),'_Vn',int2str(Vn),'_magnitude.nii']);
        data_all(:,:,:,perm) = data1-data2;
        data_all_inv(:,:,:,perm) = data2-data1;
    end

    % compute z-diff
    mean_data_all = mean(data_all,4);
    mdata_file = [char(folder),'/Vn',int2str(Vn),'_mean_',char(grpArray{1}),'-',char(grpArray{2}),'.nii'];
    mat2nii(squeeze(mean_data_all),mdata_file);
    std_data_all = std(data_all,0,4);
    sdata_file = [char(folder),'/Vn',int2str(Vn),'_std_',char(grpArray{1}),'-',char(grpArray{2}),'.nii'];
    mat2nii(squeeze(std_data_all),sdata_file);
    zdata = (data_ref-mean_data_all)./std_data_all;
    zdata_file = [char(folder),'/Vn',int2str(Vn),'_z_',char(grpArray{1}),'-',char(grpArray{2}),'.nii'];
    fprintf([pwd,'/',zdata_file,'\n']);
    mat2nii(squeeze(zdata),zdata_file);
    
    mean_data_all_inv = mean(data_all_inv,4);
    mdata_file = [char(folder),'/Vn',int2str(Vn),'_mean_',char(grpArray{2}),'-',char(grpArray{1}),'.nii'];
    mat2nii(squeeze(mean_data_all_inv),mdata_file);
    std_data_all_inv = std(data_all_inv,0,4);
    sdata_file = [char(folder),'/Vn',int2str(Vn),'_std_',char(grpArray{2}),'-',char(grpArray{1}),'.nii'];
    mat2nii(squeeze(std_data_all_inv),sdata_file);
    zdata_inv = (data_ref_inv-mean_data_all_inv)./std_data_all_inv;
    zdata_file = [char(folder),'/Vn',int2str(Vn),'_z_',char(grpArray{2}),'-',char(grpArray{1}),'.nii'];
    fprintf([pwd,'/',zdata_file,'\n']);
    mat2nii(squeeze(zdata_inv),zdata_file);
    
    % compute pval
    pval_sup =  sum(data_all>=data_ref, 4)/nperm;
    pval_sup_file = [char(folder),'/Vn',int2str(Vn),'_pval_',char(grpArray{1}),'>',char(grpArray{2}),'.nii'];
    fprintf([pwd,'/',pval_sup_file,'\n']);
    mat2nii(squeeze(pval_sup),pval_sup_file);
    pval_inf =  sum(data_all<=data_ref, 4)/nperm;
    pval_inf_file = [char(folder),'/Vn',int2str(Vn),'_pval_',char(grpArray{2}),'>',char(grpArray{1}),'.nii'];
    fprintf([pwd,'/',pval_inf_file,'\n']);
    mat2nii(squeeze(pval_inf),pval_inf_file);
    fprintf('\n');
    
end
