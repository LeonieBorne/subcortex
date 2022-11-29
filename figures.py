#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nilearn import plotting, image, masking  
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import surface
from matplotlib import gridspec
import numpy as np
import pyvista as pv
from tqdm import tqdm
import pandas as pd
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, Normalize


##############################
# HIPPOCAMPUS/SUBCORTEX MESH #
##############################

# create point cloud
# cd /home/leonie/Documents/source/gradientography-task-fmri
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

##############################
# FIGURE. GRADIENTS EIGENMAP #
##############################

resultFolder = 'result'
# resultFolder = '/home/leonie/Documents/result/PISA/Result/gradientography_nimg'

cmap = plt.get_cmap('viridis')
grp_folder = 'cohorts'
grps = ['hc', 'cc']
grps_label = ['Healthy Cohort (HC)', 'Clinical Cohort (CC)']

fig = plt.figure(figsize=(21,16), constrained_layout=False)
grid = gridspec.GridSpec(
    4, 5, left=0., right=1., bottom=0., top=1.,
    height_ratios=[0.4,1,1,1], width_ratios=[0.2,1,1,1,1],
    hspace=0.0, wspace=0.0)

i = 6
for task in ['continuing', 'naive', 'resting_state']:
    for grp in grps:
        for Vn in [2, 3]:
            # texture
            eig_file = f'{resultFolder}/tasks/{task}/{grp_folder}/{grp}/Vn{Vn}_eigenvector.nii'
            eig = image.load_img(eig_file)
            texture = surface.vol_to_surf(
                eig, mesh, interpolation='nearest', radius=3, mask_img=mask_file)
            
            # plot
            ax = fig.add_subplot(grid[i], projection='3d')
            print(f'{task} {grp} {Vn} : {max(texture):.3f}')
            plotting.plot_surf(mesh, texture, view='anterior', vmin=0, vmax=max(texture),
                                cmap=cmap, avg_method='mean',
                                axes=ax)
            ax.text(32, 0, 15, 'L', va='center', fontdict={'fontsize':30})
            ax.text(-25, 0, 15, 'R', va='center', fontdict={'fontsize':30})
            
            # variance explained
            var = pd.read_csv(f'{resultFolder}/tasks/{task}/{grp_folder}/{grp}/variance_explained.csv',
                              header=None)
            ax.text(13, 0, -30, f'{var.loc[Vn-2,0]:.2f}%', va='center', fontdict={'fontsize':30})
            i += 1
    i += 1

# add text
for label, id_grid in zip(['CONTINUING', 'NAIVE', 'BACKGROUND'], [5, 10, 15]):
    ax = fig.add_subplot(grid[id_grid])
    ax.axis('off')    
    ax.text(0.3, 0.5, label, rotation=90, 
            va='center', fontdict={'fontsize':30})

for label, id_grid in zip(grps_label, [1, 3]):
    ax = fig.add_subplot(grid[id_grid])
    ax.axis('off')
    ax.text(1, 0.5, label, ha='center', fontdict={'fontsize':30})
    ax.text(0.5, 0, 'Gradient I', ha='center', fontdict={'fontsize':30})
    ax = fig.add_subplot(grid[id_grid+1])
    ax.axis('off')
    ax.text(0.5, 0, 'Gradient II', ha='center', fontdict={'fontsize':30})

# colorbar
cax = plt.axes([1.01, 0.3, 0.03, 0.3])
cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
cbar.set_ticks([])
cbar.ax.set_title('max', fontdict={'fontsize':30}, pad=20)
cbar.ax.set_xlabel('0', fontdict={'fontsize':30}, labelpad=20)

plt.show()

################################
# FIGURE. GRADIENT I MAGNITUDE #
################################

resultFolder = 'result'
# resultFolder = '/home/leonie/Documents/result/PISA/Result/gradientography_nimg'

grp_folder = 'cohorts'
grps = ['hc', 'cc']
grps_label = ['Healthy Cohort (HC)', 'Clinical Cohort (CC)']

fontcolor = 'black'

fig = plt.figure(figsize=(21,16), constrained_layout=False, dpi=300)
grid = gridspec.GridSpec(
    4, 5, left=0., right=1., bottom=0., top=1.,
    height_ratios=[0.2,1,1,1], width_ratios=[0.2,1,1,1,1],
    hspace=0.0, wspace=0.0)

i = 6
Vn = 2
for task in ['continuing', 'naive', 'resting_state']:
    for grp in grps:
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
                if task == 'resting_state':
                    ax.text(25, 0, -30, 'anterior view', va='center', fontdict={'fontsize':30, 'color':fontcolor})
            else:
                ax.text(25, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':'white'})
                ax.text(-32, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':'white'})
                if task == 'resting_state':
                    ax.text(-25, 0, -30, 'posterior view', va='center', fontdict={'fontsize':30, 'color':'white'})
                ax.set_facecolor('lightslategrey') #(0.4,0.4,0.4))
            i += 1
    i += 1

# add text
for label, id_grid in zip(['CONTINUING', 'NAIVE', 'BACKGROUND'], [5, 10, 15]):
    ax = fig.add_subplot(grid[id_grid])
    ax.axis('off')    
    ax.text(0.3, 0.5, label, rotation=90, 
            va='center', fontdict={'fontsize':30})

for label, id_grid in zip(grps_label, [1, 3]):
    ax = fig.add_subplot(grid[id_grid])
    ax.axis('off')
    ax.text(1, 0.4, label, ha='center', fontdict={'fontsize':30, 'color':fontcolor})
    ax = fig.add_subplot(grid[id_grid+1])
    ax.axis('off')

# colorbar
import matplotlib.cm as cm
cax = plt.axes([1.01, 0.3, 0.03, 0.3])
cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
cbar.set_ticks([])
cbar.ax.set_title('0.2', fontdict={'fontsize':30, 'color':fontcolor}, pad=20)
cbar.ax.set_xlabel('0', fontdict={'fontsize':30, 'color':fontcolor}, labelpad=20)

# plt.show()
fig.savefig(f'/tmp/magnitude_Vn{Vn}.png', dpi=300)

#############################
# FIGURE. COHORT COMPARISON #
#############################

resultFolder = 'result'
# resultFolder = '/home/leonie/Documents/result/PISA/Result/gradientography_nimg'

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

grp_folder = 'cohorts'
grps = ['hc', 'cc']
grps_label = ['HC', 'CC']

fontcolor = 'darkslategrey'
fontcolor='black'

cmap=plt.get_cmap('twilight_shifted')
twil = cm.get_cmap('twilight_shifted', 1000)
newcolors = np.vstack((twil(np.linspace(0, 0.3, 400)),
                        twil(np.linspace(0.4, 0.6, 400)),
                        twil(np.linspace(0.7, 1, 400))))
cmap = ListedColormap(newcolors, name='twilight_shifted_threshold')

fig = plt.figure(figsize=(11,14), constrained_layout=False, dpi=300)
grid = gridspec.GridSpec(
    4, 3, left=0., right=1., bottom=0., top=1.,
    height_ratios=[0.2,1,1,1], width_ratios=[0.2,1,1],
    hspace=0.0, wspace=0.0)

i = 4
Vn = 2
for task in ['continuing', 'naive', 'resting_state']:
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
    i += 1

# add text
for label, id_grid in zip(['CONTINUING', 'NAIVE', 'BACKGROUND'], [3, 6, 9]):
    ax = fig.add_subplot(grid[id_grid])
    ax.axis('off')    
    ax.text(0.3, 0.5, label, rotation=90, 
            va='center', fontdict={'fontsize':30})

ax = fig.add_subplot(grid[1])
ax.axis('off')    
ax.text(1, 0.4, 'Clinical Cohort (CC) - Healthy Cohort (HC)', ha='center', 
        fontdict={'fontsize':30, 'color':fontcolor})

# colorbar
cax = plt.axes([1.07, 0.3, 0.05, 0.3])
cbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=-vmax, vmax=vmax), cmap=cmap), cax=cax)
cbar.set_ticks([-2,2])
cbar.set_ticklabels([r'-2$\sigma$',r'2$\sigma$'])
cbar.ax.tick_params(labelsize=30, labelcolor=fontcolor)
cbar.ax.set_title(f'{grps_label[1]}>{grps_label[0]}', fontdict={'fontsize':30, 'color':fontcolor}, pad=20)
cbar.ax.set_xlabel(f'{grps_label[0]}>{grps_label[1]}', fontdict={'fontsize':30, 'color':fontcolor}, labelpad=20)

plt.show()
fig.savefig(f'/tmp/cohort_comparison_Vn{Vn}.png', dpi=300)


###########################
# FIGURE. TASK COMPARISON #
###########################

resultFolder = 'result'
# resultFolder = '/home/leonie/Documents/result/PISA/Result/gradientography_TSTAT_with_confounds'

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

from matplotlib import cm
from matplotlib.colors import ListedColormap
twil = cm.get_cmap('twilight_shifted', 1000)
newcolors = np.vstack((twil(np.linspace(0, 0.3, 400)),
                        twil(np.linspace(0.4, 0.6, 400)),
                        twil(np.linspace(0.7, 1, 400))))
cmap_comp = ListedColormap(newcolors, name='twilight_shifted_threshold')

fig = plt.figure(figsize=(21,6), constrained_layout=False, dpi=300)
grid = gridspec.GridSpec(
    2, 5, left=0., right=1., bottom=0., top=1.,
    height_ratios=[0.2,1], width_ratios=[0.2,1,1,1,1],
    hspace=0.0, wspace=0.0)

i = 6
for grp in ['hc', 'cc']:
    # texture
    mag_file = f'{resultFolder}/cohort/{grp}/tasks/Vn{Vn}_z_continuing-naive.nii'
    mag = image.load_img(mag_file)
    texture = surface.vol_to_surf(
        mag, mesh, interpolation=interpolation, radius=radius,
        kind=kind, mask_img=mask_file)
    vmax = 6
    for view in ['anterior', 'posterior']:
        # plot
        ax = fig.add_subplot(grid[i], projection='3d')
        plotting.plot_surf(mesh, texture, view=view, vmin=-vmax, vmax=vmax,
                            cmap=cmap_comp, avg_method=avg_method,
                            axes=ax)
        if view == 'anterior':
            ax.text(32, 0, 15, 'L', va='center', fontdict={'fontsize':30})
            ax.text(-25, 0, 15, 'R', va='center', fontdict={'fontsize':30})
            ax.text(25, 0, -30, 'anterior view', va='center', fontdict={'fontsize':30})
        else:
            ax.text(25, 0, 15, 'L', va='center', fontdict={'fontsize':30, 'color':'white'})
            ax.text(-32, 0, 15, 'R', va='center', fontdict={'fontsize':30, 'color':'white'})
            ax.set_facecolor('lightslategrey')
            ax.text(-25, 0, -30, 'posterior view', va='center', fontdict={'fontsize':30, 'color':'white'})
        i += 1

# add text
ax = fig.add_subplot(grid[5])
ax.axis('off')
ax.text(0.3, 0.5, 'CONT - NAIVE', rotation=90, 
        va='center', fontdict={'fontsize':30})
ax = fig.add_subplot(grid[1])
ax.axis('off')
ax.text(1, 0.4, 'Healthy Cohort', ha='center', fontdict={'fontsize':30})
ax = fig.add_subplot(grid[2])
ax.axis('off')
ax = fig.add_subplot(grid[3])
ax.axis('off')
ax.text(1, 0.4, 'Clinical Cohort', ha='center', fontdict={'fontsize':30})
ax = fig.add_subplot(grid[4])
ax.axis('off')

# colorbar
cax = plt.axes([1.08, 0.2, 0.03, 0.4])
cbar = fig.colorbar(cm.ScalarMappable(norm=Normalize(vmin=-vmax, vmax=vmax), cmap=cmap_comp), cax=cax)
cbar.set_ticks([-2,2])
cbar.set_ticklabels([r'-2$\sigma$',r'2$\sigma$'])
cbar.ax.tick_params(labelsize=30)
cbar.ax.set_title('CONT>NAIVE', fontdict={'fontsize':30}, pad=20)
cbar.ax.set_xlabel('NAIVE>CONT', fontdict={'fontsize':30}, labelpad=20)

plt.show()
fig.savefig(f'/tmp/task_comparison_Vn{Vn}.png', bbox_inches='tight', dpi=300)

###############################################
# FIGURE. CORTICAL PROJECTION - HC CONTINUING #
###############################################

# parameters
resultFolder = 'result'
# resultFolder = '/home/leonie/Documents/result/PISA/Result/gradientography_age_match' ###

cmap = plt.get_cmap('viridis')
Vn = 2
grp = 'hc' ###
task = 'continuing' ###
hemi = 'left'
hip_side = 'left'
hip = 'hippocampus_left.nii' #'hippocampus_left.nii', 'hippocampus_right.nii', 'hippocampus.nii']:

# load masks
hip_file = f'masks/{hip}'
hip_msk = image.load_img(hip_file)
sub_msk_cropped = image.load_img('masks/subcortex_mask_part1_cropped.nii')
sub_msk_mni = image.load_img('masks/subcortex_mask_part1.nii')
fsaverage = datasets.fetch_surf_fsaverage()

# create figure
fig = plt.figure(figsize=(15,6), constrained_layout=False)
grid = gridspec.GridSpec(
    1, 3, left=0., right=1., bottom=0., top=1.,
    height_ratios=[1.0], width_ratios=[0.6,1,1],
    hspace=0.0, wspace=0.0)

interpolation='linear'
kind='ball'
radius=3
n_samples = None
mask_img = 'masks/cortex.nii'
row = 0
cmap = cmap
# cmap = plt.get_cmap('bone_r')

img_file = f'{resultFolder}/projection/{grp}/{task}/Vn{Vn}eigenvector_projection_{hip}'
eig_file = f'{resultFolder}/projection/{grp}/{task}/Vn{Vn}_eigenvector.nii'
# img_file = f'{resultFolder}/projection/boot/{task}/1_eigenvector_projection_{hip[:-4]}_fix.nii'
# eig_file = f'{resultFolder}/projection/boot/{task}/1_Vn{Vn}_eigenvector.nii'
img = image.load_img(img_file)
eig = image.load_img(eig_file)
eig_1d = masking.apply_mask(eig, sub_msk_cropped)
eig_mni = masking.unmask(eig_1d, sub_msk_mni)
eig_hip_1d = masking.apply_mask(eig_mni, hip_msk)
eig_hip = masking.unmask(eig_hip_1d, hip_msk)
vmin = min(eig_hip_1d)
vmax = max(eig_hip_1d)
print(f'vmin={vmin}, vmax={vmax}')
mesh_file = '/home/leonie/Documents/result/PISA/Result/gradientography_nimg/projection/102311.L.inflated.32k_fs_LR.surf.gii'
if hemi == 'left':
    pial_mesh = mesh_file#fsaverage.pial_left
    infl_mesh = mesh_file#fsaverage.infl_left
else:
    pial_mesh = mesh_file#fsaverage.pial_right
    infl_mesh = mesh_file#fsaverage.infl_right

if hip_side == 'left':
    cut_coords=[-22]
else:
    cut_coords=[24]
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
    ax = fig.add_subplot(grid[col], projection='3d')
    plotting.plot_surf(
        infl_mesh, texture, hemi=hemi, view=view,
        colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax,
        axes=ax)
    ax.dist = 7
# plot hippocampus
ax = fig.add_subplot(grid[0])
disp = plotting.plot_img(
    eig_hip, display_mode='x', threshold=0, cmap=cmap, 
    vmin=vmin, vmax=vmax,
    axes=ax, cut_coords=cut_coords, colorbar=False, annotate=False)
disp.annotate(size=25)

# colorbar
cax = plt.axes([1.05, 0.32, 0.03, 0.4])
cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), cax=cax)
cbar.set_ticks([])
cbar.ax.set_title('max', fontdict={'fontsize':30}, pad=20)
cbar.ax.set_xlabel('min', fontdict={'fontsize':30}, labelpad=20)

plt.show()

fig.savefig(f'/tmp/cortical_projection_{Vn}.png', bbox_inches='tight', dpi=300)
