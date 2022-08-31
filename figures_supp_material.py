#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 10:45:28 2021

@author: leonie
"""

from nilearn import plotting, image, masking  
# import nilearn.plotting.cm as cm
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import surface
from matplotlib import gridspec
import glob
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from matplotlib.colors import ListedColormap, Normalize

fsaverage = datasets.fetch_surf_fsaverage()
cmap = plt.get_cmap('viridis')
resultFolder = '/home/leonie/Documents/result/PISA/Result/gradientography_age_match'

####################################
# GRADIENT I - CORTICAL PROJECTION #
####################################

Vn = 2
hip = 'hippocampus_left_fix.nii' #'hippocampus_left_fix.nii', 'hippocampus_right_fix.nii', 'hippocampus_fix.nii']:
hip_file = f'/home/leonie/Documents/source/yetianmed/subcortex/masks/{hip}'
hip_msk = image.load_img(hip_file)
sub_msk_cropped = image.load_img(f'/home/leonie/Documents/source/yetianmed/subcortex/masks/subcortex_mask_part1_cropped.nii')
sub_msk_mni = image.load_img(f'/home/leonie/Documents/source/yetianmed/subcortex/masks/subcortex_mask_part1_fix.nii')

fig = plt.figure(figsize=(15,20), constrained_layout=False)
grid = gridspec.GridSpec(
    5, 5, left=0., right=1., bottom=0., top=1.,
    height_ratios=[1,1,1,1,1], width_ratios=[0.3,0.5,1,1,0.3],
    hspace=0.0, wspace=0.0)

def new_row(fig, row, grp, task, downsampled=False):
    interpolation='linear'
    kind='ball'
    radius=3
    n_samples = None
    mask_img = f'/home/leonie/Documents/source/yetianmed/subcortex/masks/cortex.nii'

    # EIGEN MAP   
    if downsampled:
        img_file = f'{resultFolder}/projection/boot/{task}/2_eigenvector_projection_{hip}'
        eig_file = f'{resultFolder}/projection/boot/{task}/2_Vn2_eigenvector.nii'
    else:
        if grp == 'hc':
            img_file = f'{resultFolder}/projection/boot/{task}/1_eigenvector_projection_{hip}'
            eig_file = f'{resultFolder}/projection/boot/{task}/1_Vn2_eigenvector.nii'
        else:
            img_file = f'{resultFolder}/projection/boot/{task}/0_eigenvector_projection_{hip}'
            eig_file = f'{resultFolder}/projection/boot/{task}/0_Vn2_eigenvector.nii'
    # if grp == 'hcc':
    #     img_file = f'{resultFolder}/tasks/{task}/cohort/{grp}/Vn{Vn}_eigenvector_projection_rand_{hip}'
    # else:
    #     img_file = f'{resultFolder}/tasks/{task}/cohort/{grp}/Vn{Vn}_eigenvector_projection_{hip}'
    img = image.load_img(img_file)
    # eig_file = f'{resultFolder}/tasks/{task}/cohort/{grp}/Vn{Vn}_eigenvector_mni.nii'
    eig = image.load_img(eig_file)
    eig_1d = masking.apply_mask(eig, sub_msk_cropped)
    eig_mni = masking.unmask(eig_1d, sub_msk_mni)
    eig_hip_1d = masking.apply_mask(eig_mni, hip_msk)
    eig_hip = masking.unmask(eig_hip_1d, hip_msk)
    vmin = min(eig_hip_1d)
    vmax = max(eig_hip_1d)
    pial_mesh = fsaverage.pial_left
    infl_mesh = fsaverage.infl_left
    hemi='left'
    cut_coords=[-22]
    texture = surface.vol_to_surf(
        img, pial_mesh, interpolation=interpolation, 
        mask_img=mask_img, kind=kind, radius=radius, n_samples=n_samples)
    r = 30
    new_text = surface.vol_to_surf(
        img, pial_mesh, interpolation=interpolation, 
        mask_img=mask_img, kind=kind, radius=r, n_samples=n_samples)
    texture[texture != texture] = new_text[texture != texture]
    
    for col in [1, 2]:
        if col == 1:
            view = 'lateral'
        else:
            view = 'medial'
        # ax = fig.add_subplot(2, 3, row*3+col+1, projection='3d')
        ax = fig.add_subplot(grid[row*5+col+1], projection='3d')
        plotting.plot_surf(
            infl_mesh, texture, hemi=hemi, view=view,
            colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
            axes=ax)
        ax.dist = 7
    # plot hippocampus
    ax = fig.add_subplot(grid[row*5+1])
    disp = plotting.plot_img(
        eig_hip, display_mode='x', threshold=0, cmap=cmap, 
        vmin=vmin, vmax=vmax,
        axes=ax, cut_coords=cut_coords, colorbar=False, annotate=False)
    disp.annotate(size=25)
    # add titles
    ax = fig.add_subplot(grid[row*5])
    ax.axis('off')
    if downsampled:
        gr = 'HC (N=31)'
    else:
        gr = 'HC (N=133)' if grp == 'hc' else 'CC (N=31)'
    ax.text(0, 0.5, f'{gr}\n {task} task', rotation=90, ha='center', va='center', fontdict={'fontsize':30})
    return fig

row = 0
for grp in ['hc', 'cc']:
    for task in ['continuing','naive']:
        fig = new_row(fig, row, grp, task, downsampled=False)
        row += 1
fig = new_row(fig, row, grp='hc', task='continuing', downsampled=True)

# colorbar
for pos in [[0.95, 0.2*i+0.05, 0.03, 0.09] for i in range(5)]:
    cax = plt.axes(pos)
    cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
    cbar.set_ticks([])
    cbar.ax.set_title('max', fontdict={'fontsize':30}, pad=20)
    cbar.ax.set_xlabel('min', fontdict={'fontsize':30}, labelpad=20)

plt.show()

fig.savefig(f'/tmp/supp_cortical_projection_{Vn}.png', bbox_inches='tight', dpi=300)

##############################
# HIPPOCAMPUS/SUBCORTEX MESH #
##############################
import pyvista as pv
# create point cloud
mask_file = 'masks/hippocampus_cropped.nii'
# mask_file = 'masks/subcortex_mask_part1_cropped.nii'

msk = image.load_img(mask_file)
msk_data = msk.get_fdata()
affine = msk.affine

xlist, ylist, zlist = [], [], []
for x in tqdm(range(msk_data.shape[0])):
    for y in range(msk_data.shape[1]):
        for z in range(msk_data.shape[2]):
            if msk_data[x,y,z] == 1:
                select = False
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        for k in [-1, 0, 1]:
                            if x+i < msk_data.shape[0] and x+i >= 0 and \
                               y+j < msk_data.shape[1] and y+j >= 0 and \
                               z+k < msk_data.shape[2] and z+k >= 0:
                                if msk_data[x+i,y+j,z+k] != 1:
                                    select = True
                            else:
                                select = True
                if select:
                    (xa, ya, za) = image.coord_transform(x, y, z, affine)
                    xlist.append(xa)
                    ylist.append(ya)
                    zlist.append(za)
points = np.array([xlist, ylist, zlist]).T

# convert point cloud in surface
cloud = pv.PolyData(points)
volume = cloud.delaunay_3d(alpha=3)
shell = volume.extract_geometry()
smooth = shell.smooth(n_iter=100, relaxation_factor=0.01,
                      feature_smoothing=False, 
                      boundary_smoothing=True,
                      edge_angle=100, feature_angle=100)

# extract faces
faces = []
i, offset = 0, 0
cc = smooth.faces 
while offset < len(cc):
    nn = cc[offset]
    faces.append(cc[offset+1:offset+1+nn])
    offset += nn + 1
    i += 1

# convert to triangles
triangles = []
for face in faces:
    if len(face) == 3:
        triangles.append(face)
    elif len(face) == 4:
        triangles.append(face[:3])
        triangles.append(face[-3:])
    else:
        print(len(face))

# create mesh
mesh = [smooth.points, np.array(triangles)]


############
# EIGENMAP #
############

import pandas as pd

resultFolder = '/home/leonie/Documents/result/PISA/Result/gradientography_TSTAT_with_confounds'

for grp_folder, grps, grps_label in zip(['cohort', 'amyloid', 'apoe', 'risk'],
                                        [['hc', 'cc'], ['neg', 'pos'], ['noAPOEe4', 'APOEe4'], ['low', 'high']],
                                        [['Healthy Cohort (HC)', 'Clinical Cohort (CC)'],
                                          [r'A$\beta$ negative', r'A$\beta$ positive'],
                                          [r'APOE $\epsilon$4 non-carriers', r'APOE $\epsilon$4 carriers'],
                                          ['Low risk group', 'High risk group']]):

    fig = plt.figure(figsize=(21,12), constrained_layout=False)
    grid = gridspec.GridSpec(
        3, 5, left=0., right=1., bottom=0., top=1.,
        height_ratios=[0.4,1,1], width_ratios=[0.2,1,1,1,1],
        hspace=0.0, wspace=0.0)
    
    i = 6
    for grp in grps:
        for task in ['naive', 'continuing']:
            for Vn in [2, 3]:
                # texture
                eig_file = f'{resultFolder}/tasks/{task}/{grp_folder}/{grp}/Vn{Vn}_eigenvector.nii'
                eig = image.load_img(eig_file)
                texture = surface.vol_to_surf(
                    eig, mesh, interpolation='nearest', radius=3, mask_img=mask_file)
                # vmax = 0.055 if Vn==2 else 0.045
                vmax = eig.get_fdata().max()
                
                # plot
                ax = fig.add_subplot(grid[i], projection='3d')
                plotting.plot_surf(mesh, texture, view='anterior', vmin=0, vmax=vmax,
                                    cmap=cmap, avg_method='mean',
                                    axes=ax)
                ax.text(32, 0, 15, 'L', va='center', fontdict={'fontsize':30})
                ax.text(-25, 0, 15, 'R', va='center', fontdict={'fontsize':30})
                
                # variance explained
                var = pd.read_csv(f'{resultFolder}/tasks/{task}/{grp_folder}/{grp}/variance_explained.csv',
                                  header=None)
                ax.text(13, 0, -30, f'{var.loc[Vn-2,0]:.1f}%', va='center', fontdict={'fontsize':30})
                i += 1
        i += 1
    
    # add text
    ax = fig.add_subplot(grid[5])
    ax.axis('off')
    ax.text(0.3, 0.5, grps_label[0], rotation=90, 
            va='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[10])
    ax.axis('off')
    ax.text(0.3, 0.5, grps_label[1], rotation=90, 
            va='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[1])
    ax.axis('off')
    ax.text(1, 0.5, 'NAIVE TASK', ha='center', fontdict={'fontsize':30})
    ax.text(0.5, 0, 'Gradient I', ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[2])
    ax.axis('off')
    ax.text(0.5, 0, 'Gradient II', ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[3])
    ax.axis('off')
    ax.text(1, 0.5, 'CONTINUING TASK', ha='center', fontdict={'fontsize':30})
    ax.text(0.5, 0, 'Gradient I', ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[4])
    ax.axis('off')
    ax.text(0.5, 0, 'Gradient II', ha='center', fontdict={'fontsize':30})
    
    # colorbar
    import matplotlib.cm as cm
    cax = plt.axes([1.01, 0.2, 0.03, 0.4])
    cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
    cbar.set_ticks([])
    cbar.ax.set_title('max', fontdict={'fontsize':30}, pad=20)
    cbar.ax.set_xlabel('0', fontdict={'fontsize':30}, labelpad=20)
    
    plt.show()
    fig.savefig(f'/tmp/eigenmaps_{grp_folder}.png', bbox_inches='tight', dpi=300)

#############
# MAGNITUDE #
#############

resultFolder = '/home/leonie/Documents/result/PISA/Result/gradientography_TSTAT_with_confounds'
fontcolor = 'black'
Vn = 2

for grp_folder, grps, grps_label in zip(['cohort', 'amyloid', 'apoe', 'risk'],
                                        [['hc', 'cc'], ['neg', 'pos'], ['noAPOEe4', 'APOEe4'], ['low', 'high']],
                                        [['Healthy Cohort (HC)', 'Clinical Cohort (CC)'],
                                         [r'A$\beta$ negative', r'A$\beta$ positive'],
                                         [r'APOE $\epsilon$4 non-carriers', r'APOE $\epsilon$4 carriers'],
                                         ['Low risk group', 'High risk group']]):

    fig = plt.figure(figsize=(21,11), constrained_layout=False)
    grid = gridspec.GridSpec(
        3, 5, left=0., right=1., bottom=0., top=1.,
        height_ratios=[0.2,1,1], width_ratios=[0.2,1,1,1,1],
        hspace=0.0, wspace=0.0)
    
    i = 6
    for grp in grps:
        for task in ['naive', 'continuing']:
            # for Vn in [2, 3]:
            # texture
            mag_file = f'{resultFolder}/tasks/{task}/{grp_folder}/{grp}/Vn{Vn}_magnitude.nii'
            # mag_file = f'{resultFolder}/tasks/{task}/cohort/Vn{Vn}_{grp}_mean_boot.nii'
            mag = image.load_img(mag_file)
            texture = surface.vol_to_surf(
                mag, mesh, interpolation='nearest', radius=3, mask_img=mask_file)
            for view in ['anterior', 'posterior']:
                # plot
                ax = fig.add_subplot(grid[i], projection='3d')
                plotting.plot_surf(mesh, texture, view=view, vmin=0, vmax=0.2,
                                    cmap=cmap, avg_method='mean',
                                    axes=ax)
                if view == 'anterior':
                    ax.text(32, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':fontcolor})
                    ax.text(-25, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':fontcolor})
                    if grp == grps[1]:
                        ax.text(25, 0, -30, 'anterior view', va='center', fontdict={'fontsize':30, 'color':fontcolor})
                else:
                    ax.text(25, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':'white'})
                    ax.text(-32, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':'white'})
                    if grp == grps[1]:
                        ax.text(-25, 0, -30, 'posterior view', va='center', fontdict={'fontsize':30, 'color':'white'})
                    ax.set_facecolor('lightslategrey') #(0.4,0.4,0.4))
                i += 1
        i += 1
    
    # add text
    ax = fig.add_subplot(grid[5])
    ax.axis('off')
    ax.text(0.3, 0.5, grps_label[0], rotation=90, 
            va='center', fontdict={'fontsize':30, 'color':fontcolor})
    ax = fig.add_subplot(grid[10])
    ax.axis('off')
    ax.text(0.3, 0.5, grps_label[1], rotation=90, 
            va='center', fontdict={'fontsize':30, 'color':fontcolor})
    ax = fig.add_subplot(grid[1])
    ax.axis('off')
    ax.text(1, 0.4, 'NAIVE TASK', ha='center', fontdict={'fontsize':30, 'color':fontcolor})
    ax = fig.add_subplot(grid[2])
    ax.axis('off')
    ax = fig.add_subplot(grid[3])
    ax.axis('off')
    ax.text(1, 0.4, 'CONTINUING TASK', ha='center', fontdict={'fontsize':30, 'color':fontcolor})
    ax = fig.add_subplot(grid[4])
    ax.axis('off')
    
    # colorbar
    import matplotlib.cm as cm
    cax = plt.axes([1.03, 0.2, 0.03, 0.4])
    cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
    # cbar.ax.tick_params(labelsize=30)
    cbar.set_ticks([])
    # cbar.set_ticklabels([0, 'max'])
    cbar.ax.set_title('0.2', fontdict={'fontsize':30, 'color':fontcolor}, pad=20)
    cbar.ax.set_xlabel('0', fontdict={'fontsize':30, 'color':fontcolor}, labelpad=20)
    
    plt.show()
    fig.savefig(f'/tmp/magnitude_{grp_folder}_Vn{Vn}.png', bbox_inches='tight', dpi=300)

###############################
# COHORT MAGNITUDE DIFFERENCE #
###############################
sort = False

# projection parameters
interpolation='nearest'
kind='line'
radius=3
def custom_function(vertices):
    val = 0
    for v in vertices:
        if abs(v) > abs(val):
            val = v
    return val
avg_method = custom_function # define function that take the max(|v|)
# avg_method = 'mean'

resultFolder = '/home/leonie/Documents/result/PISA/Result/gradientography_TSTAT_with_confounds'
fontcolor='black'
for grp_folder, grps, grps_label in zip(['cohort', 'amyloid', 'apoe', 'risk'],
                                        [['hc', 'cc'], ['neg', 'pos'], ['noAPOEe4', 'APOEe4'], ['low', 'high']],
                                        [['HC', 'CC'],
                                         [r'A$\beta$(-)', r'A$\beta$(+)'],
                                         [r'APOE $\epsilon$4(-)', r'APOE $\epsilon$4(+)'],
                                         ['Low', 'High']]):
    cmap=plt.get_cmap('twilight_shifted')
    twil = cm.get_cmap('twilight_shifted', 1000)
    newcolors = np.vstack((twil(np.linspace(0, 0.3, 400)),
                            twil(np.linspace(0.4, 0.6, 400)),
                            twil(np.linspace(0.7, 1, 400))))
    cmap = ListedColormap(newcolors, name='twilight_shifted_threshold')
    # from matplotlib import cm
    # from matplotlib.colors import ListedColormap
    # top = cm.get_cmap('Blues', 100)
    # middle = cm.get_cmap('Greys', 1000)
    # bottom = cm.get_cmap('Oranges', 100)
    # newcolors = np.vstack((top(np.linspace(1, 0.5, 400)),
    #                         middle(np.linspace(0.75, 0.25, 200)),
    #                         middle(np.linspace(0.25, 0.75, 200)),
    #                         bottom(np.linspace(0.5, 1, 400))))
    # cmap = ListedColormap(newcolors, name='OrangeBlue')
    
    
    fig = plt.figure(figsize=(21,6), constrained_layout=False)
    grid = gridspec.GridSpec(
        2, 5, left=0., right=1., bottom=0., top=1.,
        height_ratios=[0.2,1], width_ratios=[0.2,1,1,1,1],
        hspace=0.0, wspace=0.0)
    
    i = 6
    Vn = 2
    # for grp in ['hc', 'cc']:
    for task in ['naive', 'continuing']:
        # texture
        mag_file = f'{resultFolder}/tasks/{task}/{grp_folder}/Vn{Vn}_z_{grps[1]}-{grps[0]}.nii'
        mag = image.load_img(mag_file)
        texture = surface.vol_to_surf(
            mag, mesh, interpolation=interpolation, radius=radius, 
            kind=kind, mask_img=mask_file)
        vmax = 6
        for view in ['anterior', 'posterior']:
            # plot
            ax = fig.add_subplot(grid[i], projection='3d')
            plotting.plot_surf(mesh, texture, view=view, vmin=-vmax, vmax=vmax,
                                cmap=cmap, avg_method=avg_method,
                                axes=ax)
            if view == 'anterior':
                ax.text(32, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':fontcolor})
                ax.text(-25, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':fontcolor})
                ax.text(25, 0, -30, 'anterior view', va='center', fontdict={'fontsize':30, 'color':fontcolor})     
            else:
                ax.text(25, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':'white'})
                ax.text(-32, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':'white'})
                ax.set_facecolor('lightslategrey')
                ax.text(-25, 0, -30, 'posterior view', va='center', fontdict={'fontsize':30, 'color':'white'})
            i += 1
    
    # add text
    ax = fig.add_subplot(grid[5])
    ax.axis('off')
    ax.text(0.3, 0.5, f'{grps_label[1]} - {grps_label[0]}', rotation=90, 
            va='center', fontdict={'fontsize':30, 'color':fontcolor})
    ax = fig.add_subplot(grid[1])
    ax.axis('off')
    ax.text(1, 0.4, 'NAIVE TASK', ha='center', fontdict={'fontsize':30, 'color':fontcolor})
    ax = fig.add_subplot(grid[2])
    ax.axis('off')
    ax = fig.add_subplot(grid[3])
    ax.axis('off')
    ax.text(1, 0.4, 'CONTINUING TASK', ha='center', fontdict={'fontsize':30, 'color':fontcolor})
    ax = fig.add_subplot(grid[4])
    ax.axis('off')
    
    # colorbar
    cax = plt.axes([1.08, 0.2, 0.03, 0.4])
    cbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=-vmax, vmax=vmax), cmap=cmap), cax=cax)
    cbar.set_ticks([-2,2])
    cbar.set_ticklabels([r'-2$\sigma$',r'2$\sigma$'])
    cbar.ax.tick_params(labelsize=30, labelcolor=fontcolor)
    if grp_folder != 'apoe':
        cbar.ax.set_title(f'{grps_label[1]}>{grps_label[0]}', fontdict={'fontsize':30, 'color':fontcolor}, pad=20)
        cbar.ax.set_xlabel(f'{grps_label[0]}>{grps_label[1]}', fontdict={'fontsize':30, 'color':fontcolor}, labelpad=20)
    else:
        cbar.ax.set_title(f'{grps_label[1]}\n>{grps_label[0]}', fontdict={'fontsize':30, 'color':fontcolor}, pad=20)
        cbar.ax.set_xlabel(f'{grps_label[0]}\n>{grps_label[1]}', fontdict={'fontsize':30, 'color':fontcolor}, labelpad=20)
    plt.show()
    fig.savefig(f'/tmp/cohort_diff_{grp_folder}_Vn{Vn}.png', bbox_inches='tight', dpi=300)
