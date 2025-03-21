import argparse
import nibabel as nib
import numpy as np

def zero_pad_nifti_image(input_image_path, output_image_path, image_size_x, image_size_y, image_size_z):
    # Load the NIfTI image
    img = nib.load(input_image_path)

    # Get the current data and affine
    data = img.get_fdata()
    affine = img.affine

    # Calculate the padding size
    target_shape = (image_size_x, image_size_y, image_size_z)
    pad_size = [(int((target_shape[i]-data.shape[i])/2), int((target_shape[i]-data.shape[i])/2)) for i in range(3)]

    # Zero-pad the data
    padded_data = np.pad(data, pad_size, mode='constant')

    # Create a new NIfTI image with the padded data
    padded_img = nib.Nifti1Image(padded_data, affine)

    # Save the padded NIfTI image with the same filename
    nib.save(padded_img, output_image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-pad a NIfTI image to specified dimensions.")
    parser.add_argument("--image", required=True, help="Path to the input NIfTI image.")
    parser.add_argument("--image_size_x", required=True, help="Size of x-dim of the image.")
    parser.add_argument("--image_size_y", required=True, help="Size of y-dim of the image.")
    parser.add_argument("--image_size_z", required=True, help="Size of z-dim of the image.")
    args = parser.parse_args()

    input_image_path = args.image
    image_size_x = int(args.image_size_x)
    image_size_y = int(args.image_size_y)
    image_size_z = int(args.image_size_z)
    output_image_path = input_image_path

    zero_pad_nifti_image(input_image_path, output_image_path, image_size_x, image_size_y, image_size_z)
    print(f"Image {input_image_path} zero-padded and saved as {output_image_path}.")
