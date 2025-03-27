import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import torch
from os.path import join 
import matplotlib.gridspec as gridspec
from numpy import inf

def show_slices(slices, labels, figname):
    """ Function to display row of image slices """
    assert len(slices) == len(labels)
    num_rows = len(slices)
    num_cols = len(slices[0])
    plt.close()
    fig = plt.figure(figsize=(16, 12))
    fig.set_facecolor('black')  # set the background color to black
    fig.subplots_adjust(left=0.15)
    # Set up grid layout for subplots
    gs = gridspec.GridSpec(nrows=num_rows, ncols=num_cols, figure=fig, wspace=0.025, hspace=0.05)

    # Add axes for each slice
    axes = []
    # for rpws
    for i in range(num_rows):
        row_axes = []
        # Add label to first subplot in each row
        label_ax = fig.add_subplot(gs[i, 0])
        label_ax.axis('off')
        label_ax.text(-0.5, 0.5, f"{labels[i]}", ha='center', va='center', rotation=90, fontsize=14, color='white')
        for j in range(num_cols):
            ax = fig.add_subplot(gs[i, j])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            ax.imshow(slices[i][j].T, cmap="gray", origin="lower")
            row_axes.append(ax)
        axes.append(row_axes)
    fig.tight_layout()
    plt.savefig(figname, dpi=500)#,bbox_inches='tight', bbox_extra_artists=[ax], pad_inches=(0, 1, 0, 0.2))

    return fig

# general path to data
path = join('/projects','crunchie','Jan','Daten','DataSuperresolution','sub-tle001','ses-preop')

"""
# get LR and GT volumes for both contrasts
LR_input1 = join(path,'sub-tle001_ses-preop_acq-mwiVibeA10_echo-1_T2starw_LR.nii.gz')
LR_input2 = join(path,'sub-tle001_ses-preop_acq-mpmFlash1_echo-1_flip-6_mt-on_MPM_LR.nii.gz')
GT_input1 = join(path,'sub-tle001_ses-preop_acq-mwiVibeA10_echo-1_T2starw.nii.gz')
GT_input2 = join(path,'sub-tle001_ses-preop_acq-mpmFlash1_echo-1_flip-6_mt-on_MPM.nii.gz')

# get results for two-head INR
mlp_ct1 = join('~','multi_contrast_inr','runs','MR-Data_images','MR-Data_subid-sub-tle001_ct1LR-MPM_LR_ct2LR-T2starw_LR_s_12_shuf_True__FF_256_4.0_1.0__MLP2__NUML_4_N_1024_D_0.0__MSELoss__1.0__1.0__0.005_0.471_0.0008_False_32_0.7_0.1__Adam_0.0004__e49__ct1.nii.gz')
mlp_ct2 = join('~','multi_contrast_inr','runs','MR-Data_images','MR-Data_subid-sub-tle001_ct1LR-MPM_LR_ct2LR-T2starw_LR_s_12_shuf_True__FF_256_4.0_1.0__MLP2__NUML_4_N_1024_D_0.0__MSELoss__1.0__1.0__0.005_0.471_0.0008_False_32_0.7_0.1__Adam_0.0004__e49__ct2.nii.gz')

# get LR and GT volumes for both contrasts
LR_input1 = join(path,'sub-tle002__ses-preop_T2starw_MWI-MCR_4FA__R1map_LR.nii.gz')
LR_input2 = join(path,'sub-tle002__ses-preop_T2starw_MWI-MCR_4FA__R2starmap_LR.nii.gz')
GT_input1 = join(path,'sub-tle002__ses-preop_T2starw_MWI-MCR_4FA__R1map.nii.gz')
GT_input2 = join(path,'sub-tle002__ses-preop_T2starw_MWI-MCR_4FA__R2starmap.nii.gz')

# get results for two-head INR
mlp_ct1 = join('~','multi_contrast_inr','runs','MR-Data_images','MR-Data_subid-sub-tle001_ct1LR-FA_LR_ct2LR-MD_LR_s_12_shuf_True__FF_256_4.0_1.0__MLP2__NUML_4_N_1024_D_0.0__MSELoss__1.0__1.0__0.005_0.471_0.0008_False_32_0.7_0.1__Adam_0.0004__e49__ct1.nii.gz')
mlp_ct2 = join('~','multi_contrast_inr','runs','MR-Data_images','MR-Data_subid-sub-tle001_ct1LR-FA_LR_ct2LR-MD_LR_s_12_shuf_True__FF_256_4.0_1.0__MLP2__NUML_4_N_1024_D_0.0__MSELoss__1.0__1.0__0.005_0.471_0.0008_False_32_0.7_0.1__Adam_0.0004__e49__ct2.nii.gz')

# get LR and GT volumes for both contrasts
LR_input1 = join(path,'SMI_fmap_LR.nii.gz')
LR_input2 = join(path,'SMI_fw_LR.nii.gz')
GT_input1 = join(path,'SMI_fmap.nii.gz')
GT_input2 = join(path,'SMI_fw.nii.gz')
"""

# get LR and GT volumes for both contrasts
LR_input1 = join(path,'SMI_fmap_LR.nii.gz')
LR_input2 = join(path,'SMI_MD_LR.nii.gz')
GT_input1 = join(path,'SMI_fmap.nii.gz')
GT_input2 = join(path,'SMI_MD.nii.gz')

# load image data from paths
#mask = nib.load(mask).get_fdata()

img_lr1 = nib.load(LR_input1).get_fdata()
img_lr2 = nib.load(LR_input2).get_fdata()
img_gt1 = nib.load(GT_input1).get_fdata()
img_gt2 = nib.load(GT_input2).get_fdata()

img_gt1[img_gt1 == inf] = 0

x_dim, y_dim, z_dim = img_gt1.shape

# remove potential NaN values
img_lr1 = np.resize(img_lr1[~np.isnan(img_lr1)], (x_dim, y_dim, z_dim))
img_lr2 = np.resize(img_lr2[~np.isnan(img_lr2)], (x_dim, y_dim, z_dim))
img_gt1 = np.resize(img_gt1[~np.isnan(img_gt1)], (x_dim, y_dim, z_dim))
img_gt2 = np.resize(img_gt2[~np.isnan(img_gt2)], (x_dim, y_dim, z_dim))

# check for potential NaN values
if np.isnan(img_lr1).any():
    print('NaN values in the first LR image!!')
if np.isnan(img_lr2).any():
    print('NaN values in the second LR image!!')
if np.isnan(img_gt1).any():
    print('NaN values in the first HR image!!')
if np.isnan(img_gt2).any():
    print('NaN values in the second HR image!!')    

# GT
gt_sag1 = img_gt1[int(x_dim/2), :, :]
gt_cor1 = img_gt1[:, int(y_dim/2), :]
gt_ax1  = img_gt1[:, :, int(z_dim/2)]

gt_sag2 = img_gt2[int(x_dim/2), :, :]
gt_cor2 = img_gt2[:, int(y_dim/2), :]
gt_ax2  = img_gt2[:, :, int(z_dim/2)]

slice_gt = [gt_ax1, gt_cor1, gt_sag1, gt_ax2, gt_cor2, gt_sag2]

## LR
lr_sag1 = img_lr1[int(img_lr1.shape[0]/2)+1, :, :]
lr_cor1 = img_lr1[:, int(img_lr1.shape[1]/2)+1, :]
lr_ax1  = img_lr1[:, :, int(img_lr1.shape[2]/2)+1]

lr_sag2 = img_lr2[int(img_lr2.shape[0]/2)+1, :, :]
lr_cor2 = img_lr2[:, int(img_lr2.shape[1]/2)+1, :]
lr_ax2  = img_lr2[:, :, int(img_lr2.shape[2]/2)+1]

slice_lr = [lr_ax1, lr_cor1, lr_sag1, lr_ax2, lr_cor2, lr_sag2]

show_slices([slice_gt], ['High Res'], join('figures','MR-data_InputsHR.png'))
show_slices([slice_lr], ['Low Res'], join('figures','MR-data_InputsLR.png'))