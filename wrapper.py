from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import nibabel as nib
import numpy as np
import cv2
from PIL import Image

class OASIS(Dataset):
    """
    A customized data loader for OASIS.
    """
    def __init__(self, filename_label_list, input_size = 299, mean_pixel_threshold=0.7, discard_front_proportion=0.3):
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
        #resize to 360 and then crop to 299, which is the input size for inception
        self.transform = transforms.Compose([transforms.Resize(input_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5,0.5,0.5],[0.25,0.25,0.25])])
                                            

        for filename, label in filename_label_list:
            OASIS.select_slices(self, filename, label)

    def select_slices(self, filename, label):
        img = OASIS.get_image(filename)
        
        if len(img.shape) != 3:
            return
        
        img = OASIS.crop_image(img)

        slice_indices = OASIS.filter_slices(self, img)
        num_slices = min(15, len(slice_indices))
        slice_indices = np.random.choice(slice_indices, num_slices)

        for idx in slice_indices:
            self.slices.append(img[:,:,idx])
            self.labels.append(label)

    @staticmethod
    def get_image(path):
        img = nib.load(path).get_data()
        # is this the rescaling?
        return (255.0 / img.max() * img).astype(np.uint8)

    @staticmethod
    def crop_image(img):
        width, height, depth  = img.shape
        # return unchanged if square
        if width == height:
            return img
        else:
            #calculate the margin and then the two offsets
            margin = abs(width - height)
            front_offset = margin // 2
            back_offset = margin - front_offset
            if width > height:
                return img[front_offset:-back_offset,:,:]
            else:
                return img[:,front_offset:-back_offset,:]

    @staticmethod
    def filter_slices(self, img):
        img_means = img.reshape(-1, img.shape[2]).mean(axis=0)
        # zero out first discard_front_proportion
        front_index = int(self.discard_front_proportion * img_means.shape[0])
        img_means[:front_index] = 0
        threshold = img_means.max() * self.mean_pixel_threshold
        return np.arange(img.shape[2])[(img_means > threshold)]

    @staticmethod
    def resize_and_scale_image(img):
        # resize to (299,299)
        rsz = cv2.resize(img, dsize = (299,299))
        # rescale to 360
        transforms.Resize(360)
        return img

    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        # If dataset is preloaded
        img_slice = self.slices[index]
        # convert img_slice to 3-channel
        img_slice = np.dstack([img_slice] * 3)

        img_slice = Image.fromarray(img_slice)
        img_slice = self.transform(img_slice)

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
