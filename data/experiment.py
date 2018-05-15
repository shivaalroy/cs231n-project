import sys
import os
import path
import nibabel as nib
import numpy as np
from PIL import Image

DATA = 'scans/'

def preview_scan(path, save_path):
    img = nib.load(path).get_data()
    img = (255.0 / img.max() * img).astype(np.uint8)
    for i in xrange(img.shape[2]):
        im = Image.fromarray(img[:,:,i])
        im.save(save_path + str(i) + '.png')

def list_experiments():
    pass

def main():
    pass

if __name__ == '__main__':
    preview_scan('scans/OAS30001_MR_d0129/anat1/sub-OAS30001_sess-d0129_acq-TSE_T2w.nii.gz', 'images/')

