from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from PIL import Image

class OASIS(Dataset):
    """
    A customized data loader for OASIS.
    """
    def __init__(self, filename_label_list, mean_pixel_threshold=0.7, discard_front_proportion=0.3):
        """ Intialize the OASIS dataset

        Args:
            - root: root directory of the dataset
            - transform: a custom transform function
            - preload: if preload the dataset into memory
        """
        self.slices = []
        self.labels = []
        self.mean_pixel_threshold = mean_pixel_threshold
        self.discard_front_proportion = discard_front_proportion

        for filename, label in filename_label_list:
            select_slices(filename, label)

    def select_slices(self, filename, label):
        # TODO
        img = OASIS.get_image(filename)
        img = OASIS.crop_image(img)
        img = OASIS.resize_and_scale_image(img)
        slice_indices = OASIS.filter_slices(img)
        for idx in slice_indices:
            # self.slices append img[idx]
            # self.labels append label

    @staticmethod
    def get_image(path):
        img = nib.load(path).get_data()
        # is this the rescaling?
        return (255.0 / img.max() * img).astype(np.uint8)

    @staticmethod
    def crop_image(img):
        # TODO
        return img

    @staticmethod
    def filter_slices(img):
        img_means = img.reshape(-1, img.shape[2]).mean(axis=0)
        # zero out first discard_front_proportion
        front_index = int(self.discard_front_proportion * img_means.shape[0])
        img_means[:front_index] = 0
        threshold = img_means.max() * self.mean_pixel_threshold
        return img_means > threshold

    @staticmethod
    def resize_and_scale_image(img):
        # TODO: figure out what to resize to
        # TODO: figure out what to rescale to
        return img

    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        # If dataset is preloaded
        img_slice = self.slices[index]
        # convert img_slice to 3-channel
        #     image = np.dstack([img[:,:,i]] * 3)

        label = self.labels[index]
        return img_slice, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return len(self.slices)

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
