from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from PIL import Image

class OASIS(Dataset):
    """
    A customized data loader for OASIS.
    """
    def __init__(self, root, filenames, transform=None, preload=False):
        """ Intialize the OASIS dataset

        Args:
            - root: root directory of the dataset
            - transform: a custom transform function
            - preload: if preload the dataset into memory
        """
        self.images = None
        self.labels = None
        self.filenames = filenames
        self.root = root
        self.transform = transform

        # if preloaded put set labels and images
        if preload:
            self.labels = []
            self.images = []
            self.len = 0
            for path, label in self.filenames:
                img = nib.load(path).get_data()
                img = (255.0 / img.max() * img).astype(np.uint8)
                for i in range(img.shape[2]):
                    self.len += 1
                    print(img.shape)
                    image = np.dstack([img[:,:,i]] * 3)
                    print(image.shape)
                    self.images.append(image)
                    self.labels.append(label)
            print('finished preloading')

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
            image = (255.0 / img.max() * img).astype(np.uint8)
            print('Not pre-loaded, before stacking')
            print(image.shape)
            image = np.dstack([image] * 3)
            print('Not pre-loaded, after stacking')
            print(image.shape)

        # May use transform function to transform samples
        # e.g., random crop, whitening
        if self.transform is not None:
            image = self.transform(image)

        # return image and label
        return image, label

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
