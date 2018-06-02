import glob
import os
import os.path as osp
import numpy as np
import re
import random
import nibabel as nib

#splits the data into train, validation, and test filenames
def split_data(root, labels_csv):
    # read in labels_csv and create dictionary
    label_array = np.genfromtxt(labels_csv, delimiter=',', dtype='str', skip_header=1)

    # create a dictionary from the ids to the labels
    label_dict = {experiment: label for experiment, label in label_array if label != ''}
    print('num labels is', len(label_dict))

    # read filenames
    all_filenames = glob.glob(osp.join(root, 'OAS*', '*', '*T2w.nii.gz'))
    all_filenames = [filename for filename in all_filenames if 'TSE' in filename]
    all_filenames = [(filename[len(root)+1:].split('/')[0], filename) for filename in all_filenames]
    print('num filenames is', len(all_filenames))

    # findings:
        # num images that are square 1433
        # if img.shape[0] != img.shape[1]: continue




    counter = 0
    all_experiments = {}
    for experiment, path in all_filenames:
        if experiment not in label_dict: continue
        # print(experiment, path)
        # img = nib.load(path).get_data()
        # img = (255.0 / img.max() * img).astype(np.uint8)
        # print(img.shape)
        if experiment not in all_experiments:
            all_experiments[experiment] = [path, label_dict[experiment]]
        counter += 1
        # if counter == 10: break
    print('num experiments is', len(all_experiments))

    for key in list(all_experiments.keys())[:5]:
        print(key, all_experiments[key])

    return


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