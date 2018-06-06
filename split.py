import glob
import os.path as osp
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(base_dir, labels_csv, balanced = False, train_size=0.7):
    """ Splits the data into train, validation, and test filenames"""
    # read in labels_csv and create dictionary
    label_array = np.genfromtxt(labels_csv, delimiter=',', dtype='str', skip_header=1)

    # create a dictionary from the ids to the labels
    label_mapping = {'0.0': 0, '0.5': 1, '1.0': 2}
    label_dict = {experiment: label_mapping[label] for experiment, label in label_array if label in label_mapping}
    print('num labels is', len(label_dict))

    # read filenames
    all_filenames = glob.glob(osp.join(base_dir, 'OAS*', '*', '*T2w.nii.gz'))
    all_filenames = [filename for filename in all_filenames if 'TSE' in filename]
    all_filenames = [(filename[len(base_dir)+1:].split('/')[0], filename) for filename in all_filenames]
    print('num filenames is', len(all_filenames))

    counts = [0]*3
    experiments = set()
    filename_labels = []
    for experiment, path in all_filenames:
        if experiment not in label_dict: continue
        if experiment not in experiments:
            experiments.add(experiment)
            filename_labels.append((path, label_dict[experiment]))
            counts[label_dict[experiment]] += 1
    print('num experiments is', len(filename_labels))
    print('counts per class:', counts)
    
    if balanced:
        # downsample all these labels to try to balance the labels
        labels = np.array([x[1] for x in filename_labels])
        i_class0 = np.where(labels == 0)[0]
        i_class1 = np.where(labels == 1)[0]
        i_class2 = np.where(labels == 2)[0]

        n_class0 = len(i_class0)
        n_class1 = len(i_class1)

        i_class0_downsampled = np.random.choice(i_class0, size=n_class1, replace=False)
        filenames_indices = np.hstack((i_class0_downsampled, i_class1, i_class2)) 
        filename_labels = [filename_labels[i] for i in range(len(labels)) if i in filenames_indices]
    

    train_filenames, test_filenames = train_test_split(filename_labels,
                                                        train_size=train_size,
                                                        random_state=42,
                                                        shuffle=True,
                                                        stratify=list(zip(*filename_labels))[1])
    val_filenames, test_filenames = train_test_split(test_filenames,
                                                        train_size=0.5,
                                                        random_state=42,
                                                        shuffle=True,
                                                        stratify=list(zip(*test_filenames))[1])

    return train_filenames, val_filenames, test_filenames
