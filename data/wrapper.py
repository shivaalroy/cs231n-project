import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import re
import csv
import glob
import nibabel as nib
import os.path as osp
import numpy as np
from PIL import Image

class OASIS(Dataset):
    """
    A customized data loader for OASIS.
    """
    def __init__(self, root, labels_csv, transform=None, preload=False):
        """ Intialize the OASIS dataset

        Args:
            - root: root directory of the dataset
            - tranform: a custom tranform function
            - preload: if preload the dataset into memory
        """
        self.images = None
        self.labels = None
        self.filenames = [] 
        self.root = root
        self.transform = transform

        # get labels_csv
        # check if path exists

        # read filenames
        filenames = glob.glob(osp.join(root, 'OAS*', 'anat1', '*T2w.nii.gz'))
        filenames = [x for x in filenames if 'TSE' in x]
        print(filenames[:10])
        # print(filenames)
        print(len(filenames))

        # read in labels_csv and create dictionary
        label_array = np.genfromtxt(labels_csv, delimiter=',', dtype='str', skip_header=1)
        print(label_array[:10])
        print(len(label_array))

        #create a dictionary from the ids to the labels
        label_dict = {}
        for entry in label_array:
            label_dict[entry[0]] = entry[1]

        for fn in filenames:
        #extract experiment_id from fn # 'scans/experiment_id/...'
            experiment_id = re.split('/', fn)[1]
            # check if the id has a corresponding label, then append it to the dict
            if experiment_id in label_dict:
                label = label_dict[experiment_id]
                # check if label isn't empty string
                if label:
                    self.filenames.append((fn, float(label)))
       
        #look at 10 items in the dict
        print(self.filenames[:10])
        self.len = len(self.filenames)
        print(self.len)

        # for fn in filenames:
        #   extract experiment_id from fn # 'scans/experiment_id/...'
        # for each

        #     for fn in filenames:
        #         self.filenames.append((fn, i)) # (filename, label) pair

        # self.len = len(self.filenames)

    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if self.images is not None:
            # If dataset is preloaded
            image = self.images[index]
            label = self.labels[index]
        else:
            # If on-demand data loading
            path, label = self.filenames[index]

            img = nib.load(path).get_data()
            image_array = (255.0 / img.max() * img).astype(np.uint8)
     
        # May use transform function to transform samples
        # e.g., random crop, whitening
        if self.transform is not None:
            image_array = self.transform(image_array)
            '''for i in range(image_array.shape[2]):
                image = image_array[:,:,i]
                image_array[:,:,i] = self.transform(image)'''
        # return image and label
        return image_array, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

def main():
    dataset = OASIS('scans', 'OASIS3_MRID2Label_051418.csv')

    image_array, label = dataset[3]
    save_path = 'test_preview_scan/'

    # save the images so we can look at them
    for i in range(image_array.shape[2]):
        image = Image.fromarray(image_array[:,:,i])
        image.save(save_path + str(i) + '.png')

if __name__ == '__main__':
    main()
