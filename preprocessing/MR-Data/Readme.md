### How to preprocess the MR data

Example for MD and fmap images.
...
├── sub-tle001
|   ├── ses-preop
│       ├── SMI_fmap.nii.gz (136,134,81)
│       ├── SMI_MD.nii.gz   (136,134,81)
```

### Step 1: Pad/Crop images to size cleanly dividable by 4

First we need to get the data into a size that is cleanly devidable into an integer. The downsampling factor for latter is 4, so we need to pad or crop the y und z dimension of our data. 
Example for the fmap:

`/home/janmeyer/miniconda3/envs/PyTorch/bin/python /home/janmeyer/multi_contrast_inr/preprocessing/MR-Data/pad.py --image /projects/crunchie/Jan/Daten/DataSuperresolution/sub-tle001/ses-preop/SMI_fmap.nii.gz --image_size_x 136 --image_size_y 136 --image_size_z 80`

Now your directory should look like this:

```
.
├── SMI_fmap.nii.gz (136,136,80)
└── SMI_MD.nii.gz   (136,136,80)
```

### Step 2: Clean up images

Some of the data has some pretty nasty contents such as distracting NaN, inf and unimportant negative values. These are not relevant for us and can thus be zeroed out.
Example for the fmap:

`/home/janmeyer/miniconda3/envs/PyTorch/bin/python /home/janmeyer/multi_contrast_inr/preprocessing/MR-Data/CleanData.py --image /projects/crunchie/Jan/Daten/DataSuperresolution/sub-tle001/ses-preop/SMI_fmap.nii.gz`

The content displayed in the directory should still look the same:

```
.
├── SMI_fmap.nii.gz (136,136,80)
└── SMI_MD.nii.gz   (136,136,80)
```

### Step 3: Downsampling to obtain LR images

We want to artificially downsample the images while preserving the voxel spacing. This is done for one dimension (as specified by the view) by a factor of 4 without affecting the other dimensions.
To downsample the fmap in the axial dimension run:

`/home/janmeyer/miniconda3/envs/PyTorch/bin/python /home/janmeyer/multi_contrast_inr/preprocessing/MR-Data/downsample.py --image /projects/crunchie/Jan/Daten/DataSuperresolution/sub-tle001/ses-preop/SMI_fmap.nii.gz --view axial`

Now your dataset directory should look like the following, when additionally downsampling MD in the sagittal dimension:
```

├── SMI_fmap.nii.gz     (136,136,80)
├── SMI_fmap_LR.nii.gz  (34,136,80)
├── SMI_MD.nii.gz       (136,136,80)
└── SMI_MD_LR.nii.gz    (136,136,20)
```

### Step 4: Clean up the LR scans

Lastly we also can clean up the LR images just to be sure.
Example for fmap_LR:

`/home/janmeyer/miniconda3/envs/PyTorch/bin/python /home/janmeyer/multi_contrast_inr/preprocessing/MR-Data/CleanData.py --image /projects/crunchie/Jan/Daten/DataSuperresolution/sub-tle001/ses-preop/SMI_fmap_LR.nii.gz`
