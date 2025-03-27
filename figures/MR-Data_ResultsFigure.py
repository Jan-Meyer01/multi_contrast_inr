import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from os.path import join 
import matplotlib.gridspec as gridspec
from numpy import inf


def crop_array(arr):

    # find the first and last non-zero rows
    first_row = (arr != 0).any(axis=1).argmax()
    last_row = len(arr) - (arr != 0)[::-1].any(axis=1).argmax() - 1

    # find the first and last non-zero columns
    first_col = (arr != 0).any(axis=0).argmax()
    last_col = len(arr[0]) - (arr != 0)[:,::-1].any(axis=0).argmax() - 1

    # slice the array to create a new array with only the non-zero rows and columns
    return arr[first_row:last_row+1, first_col:last_col+1]


def show_slices(slices, labels, min_error_left, max_error_left, figname):
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
            # plot images
            if i == 2:
                ax.imshow(slices[i][j].T, cmap="hot", origin="lower", vmin=min_error_left, vmax=max_error_left)   
            else:
                ax.imshow(slices[i][j].T, cmap="gray", origin="lower")
            row_axes.append(ax)
        axes.append(row_axes)
        

    #fig.tight_layout()
    plt.savefig(figname, dpi=500)#,bbox_inches='tight', bbox_extra_artists=[ax], pad_inches=(0, 1, 0, 0.2))

    return fig

# general path to data
path = join('/projects','crunchie','Jan','Daten','DataSuperresolution','sub-tle001','ses-preop')

"""
# get LR and GT volumes for both contrasts
GT_input1 = join(path,'sub-tle001_ses-preop_acq-mwiVibeA10_echo-1_T2starw.nii.gz')
GT_input2 = join(path,'sub-tle001_ses-preop_acq-mpmFlash1_echo-1_flip-6_mt-on_MPM.nii.gz')

# get results for two-head INR
mlp_ct1 = join('~','multi_contrast_inr','runs','MR-Data_images','MR-Data_subid-sub-tle001_ct1LR-MPM_LR_ct2LR-T2starw_LR_s_12_shuf_True__FF_256_4.0_1.0__MLP2__NUML_4_N_1024_D_0.0__MSELoss__1.0__1.0__0.005_0.471_0.0008_False_32_0.7_0.1__Adam_0.0004__e49__ct1.nii.gz')
mlp_ct2 = join('~','multi_contrast_inr','runs','MR-Data_images','MR-Data_subid-sub-tle001_ct1LR-MPM_LR_ct2LR-T2starw_LR_s_12_shuf_True__FF_256_4.0_1.0__MLP2__NUML_4_N_1024_D_0.0__MSELoss__1.0__1.0__0.005_0.471_0.0008_False_32_0.7_0.1__Adam_0.0004__e49__ct2.nii.gz')

# get LR and GT volumes for both contrasts
GT_input1 = join(path,'sub-tle002__ses-preop_T2starw_MWI-MCR_4FA__R1map.nii.gz')
GT_input2 = join(path,'sub-tle002__ses-preop_T2starw_MWI-MCR_4FA__R2starmap.nii.gz')

# get results for two-head INR
mlp_ct1 = join('~','multi_contrast_inr','runs','MR-Data_images','MR-Data_subid-sub-tle001_ct1LR-R2starmap_LR_ct2LR-R1map_LR_s_12_shuf_True__FF_256_4.0_1.0__MLP2__NUML_4_N_1024_D_0.0__MSELoss__1.0__1.0__0.005_0.471_0.0008_False_32_0.7_0.1__Adam_0.0004__e49__ct1.nii.gz')
mlp_ct2 = join('~','multi_contrast_inr','runs','MR-Data_images','MR-Data_subid-sub-tle001_ct1LR-R2starmap_LR_ct2LR-R1map_LR_s_12_shuf_True__FF_256_4.0_1.0__MLP2__NUML_4_N_1024_D_0.0__MSELoss__1.0__1.0__0.005_0.471_0.0008_False_32_0.7_0.1__Adam_0.0004__e49__ct2.nii.gz')
"""


# get LR and GT volumes for both contrasts
GT_input1 = join(path,'SMI_MD.nii.gz')
GT_input2 = join(path,'SMI_fmap.nii.gz')

# get results for two-head INR
mlp_ct1 = join('~','multi_contrast_inr','runs','MR-Data_images','MR-Data_subid-sub-tle001_ct1LR-fmap_LR_ct2LR-MD_LR_s_12_shuf_True__FF_256_4.0_1.0__MLP2__NUML_4_N_1024_D_0.0__MSELoss__1.0__1.0__0.005_0.471_0.0008_False_32_0.7_0.1__Adam_0.0004__e99__ct1.nii.gz')
mlp_ct2 = join('~','multi_contrast_inr','runs','MR-Data_images','MR-Data_subid-sub-tle001_ct1LR-fmap_LR_ct2LR-MD_LR_s_12_shuf_True__FF_256_4.0_1.0__MLP2__NUML_4_N_1024_D_0.0__MSELoss__1.0__1.0__0.005_0.471_0.0008_False_32_0.7_0.1__Adam_0.0004__e99__ct2.nii.gz')

# load image data from paths
#mask = nib.load(mask).get_fdata()

img_gt1 = nib.load(GT_input1).get_fdata()
img_gt2 = nib.load(GT_input2).get_fdata()

# if there are inf values in the array replace them with zero
img_gt1[img_gt1 == inf] = 0
img_gt2[img_gt2 == inf] = 0

x_dim, y_dim, z_dim = img_gt1.shape

img_ct1_mlp = nib.load(mlp_ct1).get_fdata()
img_ct2_mlp = nib.load(mlp_ct2).get_fdata()

#img_ct1_only = np.zeros(img_ct1_mlp.shape) #nib.load(CT2_only)
#img_ct2_only = np.zeros(img_ct1_mlp.shape) #nib.load(CT1_only)


# GT
gt_sag1 = img_gt1[int(x_dim/2), :, :]
gt_cor1 = img_gt1[:, int(y_dim/2), :]
gt_ax1  = img_gt1[:, :, int(z_dim/2)]

gt_sag2 = img_gt2[int(x_dim/2), :, :]
gt_cor2 = img_gt2[:, int(y_dim/2), :]
gt_ax2  = img_gt2[:, :, int(z_dim/2)]

slice_gt = [gt_ax1, gt_cor1, gt_sag1, gt_ax2, gt_cor2, gt_sag2]
#slice_gt = [crop_array(i) for i in slice_gt]

## LR
"""
lr_sag1 = img_lr1[int(x_dim/2), :, :]
lr_cor1 = img_lr1[:, int(y_dim/2), :]
lr_ax1  = img_lr1[:, :, int(z_dim/2)]

lr_sag2 = img_lr2[int(x_dim/2), :, :]
lr_cor2 = img_lr2[:, int(y_dim/2), :]
lr_ax2  = img_lr2[:, :, int(z_dim/2)]

err_lr1 = np.abs(lr_sag1-gt_sag1)
err_lr2 = np.abs(lr_sag2-gt_sag2)

slice_lr = [lr_ax1, lr_cor1, lr_sag1, err_lr1 , lr_ax2, lr_cor2, lr_sag2,err_lr2]
slice_lr = [crop_array(i) for i in slice_lr]
"""

# MLP2
mlp_sag1 = img_ct2_mlp[int(x_dim/2), :, :]
mlp_cor1 = img_ct2_mlp[:, int(y_dim/2), :]
mlp_ax1  = img_ct2_mlp[:, :, int(z_dim/2)]

mlp_sag2 = img_ct1_mlp[int(x_dim/2), :, :]
mlp_cor2 = img_ct1_mlp[:, int(y_dim/2), :]
mlp_ax2  = img_ct1_mlp[:, :, int(z_dim/2)]

slice_mlp = [mlp_ax1, mlp_cor1, mlp_sag1, mlp_ax2, mlp_cor2, mlp_sag2]
#slice_mlp = [crop_array(i) for i in slice_mlp]

err_mlp_sag1 = np.abs(gt_sag1-mlp_sag1)
err_mlp_sag2 = np.abs(gt_sag2-mlp_sag2)

err_mlp_cor1 = np.abs(gt_cor1-mlp_cor1)
err_mlp_cor2 = np.abs(gt_cor2-mlp_cor2)

err_mlp_ax1 = np.abs(gt_ax1-mlp_ax1)
err_mlp_ax2 = np.abs(gt_ax2-mlp_ax2)

slice_error = [err_mlp_ax1, err_mlp_cor1, err_mlp_sag1, err_mlp_ax2, err_mlp_cor2, err_mlp_sag2]
#slice_error = [crop_array(i) for i in slice_error]

"""
min_error_left  = min(np.min(err_lr1), np.min(err_mlp1))
min_error_right = min(np.min(err_lr2), np.min(err_mlp2))
max_error_left  = max(np.max(err_lr1), np.max(err_mlp1))
max_error_right = max(np.max(err_lr2), np.max(err_mlp2))

slices = [slice_lr, slice_mlp, slice_gt] 
labels = ['LR', 'Split-head INR', 'HR GT']
"""

# calc errors
min_error1 = min(np.min(err_mlp_sag1), np.min(err_mlp_cor1), np.min(err_mlp_ax1))
max_error1 = max(np.max(err_mlp_sag1), np.max(err_mlp_cor1), np.max(err_mlp_ax1))
#min_error2 = min(np.min(err_mlp_sag2), np.min(err_mlp_cor2), np.min(err_mlp_ax2))
#max_error2 = max(np.max(err_mlp_sag2), np.max(err_mlp_cor2), np.max(err_mlp_ax2))

slices = [slice_gt, slice_mlp, slice_error] 
labels = ['HR GT', 'Split-head INR', 'Error']

show_slices(slices, labels, min_error1, max_error1, join('figures','MR-data_sub-tle001_fmap_MD_Epoch999.png'))