import nibabel as nib
import numpy as np

def convert_nii_3d(path):
    img = nib.load(path).get_data()
    return (255.0 / img.max() * img).astype(np.uint8)