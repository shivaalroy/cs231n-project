import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os.path as osp
import numpy as np
from PIL import Image

class OASIS(Dataset):
    """
    A customized data loader for OASIS.
    """
    def __init__(self, root, transform=None, preload=False):
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

        # for fn in filenames:
        #   extract experiment_id from fn # 'scans/experiment_id/...'
        # for each

        #     for fn in filenames:
        #         self.filenames.append((fn, i)) # (filename, label) pair

        # self.len = len(self.filenames)

    # def __getitem__(self, index):
    #     """ Get a sample from the dataset
    #     """
    #     if self.images is not None:
    #         # If dataset is preloaded
    #         image = self.images[index]
    #         label = self.labels[index]
    #     else:
    #         # If on-demand data loading
    #         image_fn, label = self.filenames[index]
    #         image = Image.open(image_fn)

    #     # May use transform function to transform samples
    #     # e.g., random crop, whitening
    #     if self.transform is not None:
    #         image = self.transform(image)
    #     # return image and label
    #     return image, label

    # def __len__(self):
    #     """
    #     Total number of samples in the dataset
    #     """
    #     return self.len

def main():
    dataset = OASIS('scans')

if __name__ == '__main__':
    main()