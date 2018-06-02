import glob
import os.path as osp
import numpy as np
import re
import random

#splits the data into train, validation, and test filenames
def split_data(root, labels_csv):
    # read filenames
    all_filenames = glob.glob(osp.join(root, 'OAS*', 'anat1', '*T2w.nii.gz'))
    all_filenames = [x for x in all_filenames if 'TSE' in x]
    print('num filenames is', len(all_filenames))

    # read in labels_csv and create dictionary
    label_array = np.genfromtxt(labels_csv, delimiter=',', dtype='str', skip_header=1)
    print(label_array[:10])
    print(len(label_array))

    #create a dictionary from the ids to the labels
    label_dict = {}
    for entry in label_array:
        label_dict[entry[0]] = entry[1]

    filename_labels = [] #stores (filename, label) tuples
    for fn in all_filenames:
    #extract experiment_id from fn # 'scans/experiment_id/...'
        experiment_id = re.split('/', fn)[1]
        # check if the id has a corresponding label, then append it to the dict
        if experiment_id in label_dict:
            label = label_dict[experiment_id]
            # check if label isn't empty string
            if label:
                filename_labels.append((fn, float(label)))
    print('num files with labels is', len(filename_labels))

    random.shuffle(filename_labels)
    size = len(filename_labels)
    train_filenames = filename_labels[:int(.8*size)]
    val_filenames = filename_labels[int(.8*size):int(.9*size)]
    test_filenames = filename_labels[int(.9*size):]
    return (train_filenames, val_filenames, test_filenames)