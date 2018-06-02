import nibabel as nib
import numpy as np

def convert_nii_3d(path):
    img = nib.load(path).get_data()
    return (255.0 / img.max() * img).astype(np.uint8)

def get_trainable_slices(img, mean_pixel_threshold=0.7, discard_front_proportion=0.3):
    img_means = img.reshape(-1, img.shape[2]).mean(axis=0)
    # zero out first discard_front_proportion
    front_index = int(discard_front_proportion * img_means.shape[0])
    img_means[:front_index] = 0
    threshold = img_means.max() * mean_pixel_threshold
    return img_means > threshold