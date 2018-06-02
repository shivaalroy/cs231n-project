import sys
import os
import os.path as path
import nibabel as nib
import numpy as np
from PIL import Image

def preview_scan(path, save_path):
    img = nib.load(path).get_data()
    img = (255.0 / img.max() * img).astype(np.uint8)
    for i in range(img.shape[2]):
        im = Image.fromarray(img[:,:,i])
        im.save(save_path + str(i) + '.png')

if __name__ == '__main__':
    # preview_scan('scans/OAS30001_MR_d0129/anat1/sub-OAS30001_sess-d0129_acq-TSE_T2w.nii.gz', 'preview_scan/')
    # preview_scan('scans/OAS30001_MR_d0129/anat4/sub-OAS30001_sess-d0129_T2w.nii.gz', 'preview_scan_non_TSE/')
    # preview_scan('scans/OAS30852_MR_d3014/anat2/sub-OAS30852_sess-d3014_acq-TSE_run-02_T2w.nii.gz', 'preview_scan_TSE_run/')
    # preview_scan('scans/OAS30001_MR_d2430/anat2/sub-OAS30001_sess-d2430_acq-TSE_T2w.nii.gz', 'preview_scan_anat2_TSE/')
    preview_scan('scans/OAS30001_MR_d2430/anat2/sub-OAS30001_sess-d2430_acq-TSE_T2w.nii.gz', 'preview_scan_anat2_TSE/')

