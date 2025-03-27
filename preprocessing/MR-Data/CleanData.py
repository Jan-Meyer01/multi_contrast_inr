import nibabel as nib
import nibabel.processing as nip
import numpy as np
import argparse
from numpy import inf


def resample_and_save(input_file):
    # Load the NIfTI image
    nifti_image = nib.load(input_file)
    img = nifti_image.get_fdata()

    # if there are inf values in the array replace them with zero
    img[img == inf] = 0
    # same for NaN values
    img[np.isnan(img)] = 0
    # and zero out negative values
    img[img < 0] = 0

    # Create a new NIfTI image and save
    new_nifti_image = nib.Nifti1Image(img, nifti_image.affine, nifti_image.header)
    nib.save(new_nifti_image, input_file)
    print(f"Cleaned image saved as {input_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up a NIfTI image by removing negative, inf and NaN values.")
    parser.add_argument("--image", help="Input NIfTI image file")
    resample_and_save(input_file=parser.parse_args().image)
