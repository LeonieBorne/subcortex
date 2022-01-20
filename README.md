# Gradientography pipeline for task fMRI

Here we propose to apply gradientography to study task fMRI.
[Gradientography](https://www.nature.com/articles/s41593-020-00711-6.epdf?sharing_token=Fzk9fg_oTs49l2_4GcFHvtRgN0jAjWel9jnR3ZoTv0OcoEh_rWSSGTYcOuTVFJlvyoz7cKiJgYmHRlYIGzAnNt5tMyMZIXn3xdgdMC_wzDAONIDh5m0cUiLGzNChnEK_AHqVJl2Qrno8-hzk8CanTnXjGX3rRfZX3WXgTLew1oE%3D) is a pipeline that uses resting-state functional MRI (fMRI) to map the complex topographic organization of the human subcortex, enabling the characterization of cortico–subcortical connectivity. 
We adapted the pipeline using psychophysiological interactions (PPI) to study the task-induced changes in connectivity.

Most of the Matlab scripts described below are adaptations of the [gradientography pipeline available on github](https://github.com/yetianmed/subcortex).
Results visualizations are done in Python, using [Nilearn](https://nilearn.github.io).

## preprocessing.m

This script do a few additional preprocessing steps on top of HCP minimal preprocessing pipeline (Glasser et al 2013).

## similarity_matrix.m

This script compute the similarity matrix, using PPI, from the preprocessed fMRI scan.

## gradients.m

This script compute the gradients for each task and each group from the similarity matrices.

The result folder has the following organisation:

```
result
└───tasks
│   └───naive
│       └───cohorts
|           └───hc
│               │   subjects.txt
│               │   savg.mat
│               │   Vn2_eigenvector.nii
│               │   Vn2_magnitude.nii
│               │   ...
|           └───cc
│               │   ...
│   └───continuing
│       └───cohorts
|           └───hc
│               │   ...
|           └───cc
│               │   ...
```

## permutation.m

This script do a permutation test between cohorts (for a fixed task) or between tasks (for a given cohort).

For a cohort permutation (for the naive task), the result folder needs to have the following organisation:
```
result
└───tasks
│   └───naive
│       └───cohorts
|           └───hc
│               │   subjects.txt
|           └───cc
│               │   subjects.txt
```

For a task permutation (for the healthy cohort), the result folder needs to have the following organisation:
```
result
└───cohorts
│   └───hc
|       │   subjects.txt
```

The script will create a permutation folder in each group/task permuted. The output files are used for the visualisation made by the Python scripts below.

## subcortical_projection.py

## cortical_projection.py




