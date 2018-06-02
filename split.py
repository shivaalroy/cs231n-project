import glob
import os.path as osp
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(base_dir, labels_csv, train_size=0.7):
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

    # for filename_label in filename_labels[:5]:
    #     print(filename_label)

    # size = len(filename_labels)
    # max_train_idx = int(train_size*size)
    # max_val_idx = int((train_size+(1-train_size)/2) * size)

    # np.random.shuffle(filename_labels)
    # train_filenames = filename_labels[:max_train_idx]
    # val_filenames = filename_labels[max_train_idx:max_val_idx]
    # test_filenames = filename_labels[max_val_idx:]

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
