import copy
import glob
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.metrics import classification_report

from wrapper import OASIS
from split import split_data

scans_home = 'data/scans'
labels_file = 'data/OASIS3_MRID2Label_052918.csv'
n_classes = 3
freeze_layers = False
start_freeze_layer = 'Mixed_5d'
use_parallel = True

loss_weights = torch.tensor([1.,3.,5.])
if torch.cuda.is_available():
    loss_weights = loss_weights.cuda()
criterion = nn.CrossEntropyLoss(weight=loss_weights)
optimizer_type = torch.optim.Adam
lr_scheduler_type = optim.lr_scheduler.StepLR
num_epochs = 5
best_model_filepath = None
load_model_filepath = None
#load_model_filepath = 'model_best.pth.tar'

def get_counts(filename_labels):
    counts = [0]*3
    for filename, label in filename_labels:
        counts[label] += 1
    return counts


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, use_gpu, num_epochs=5):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    # list of models from all epochs
    model_list = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                    model = model.cuda()
                else:
                    input = Variable(inputs)
                    labels = Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                if type(outputs) == tuple:
                    outputs, _ = outputs
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.item() / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            # TODO: use a better metric than accuracy?
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

                state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                if best_model_filepath is not None:
                    torch.save(state, best_model_filepath)
        
        model_list.append(copy.deepcopy(model))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model_list, model


def evaluate_model(model, testset_loader, test_size, use_gpu):
    model.train(False)  # Set model to evaluate mode

    predictions = []
    # Iterate over data
    for inputs, labels in tqdm(testset_loader):
        # TODO: wrap them in Variable?
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # forward
        outputs = model(inputs)
        if type(outputs) == tuple:
            outputs, _ = outputs
        _, preds = torch.max(outputs.data, 1)
        predictions.extend(preds.tolist())
    return predictions


def load_saved_model(filepath, model, optimizer=None):
    state = torch.load(filepath)
    model.load_state_dict(state['state_dict'])
    # Only need to load optimizer if you are going to resume training on the model
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])


def run():
    train_filenames, val_filenames, test_filenames = split_data(scans_home, labels_file)
    print('train filenames size: ', len(train_filenames))
    print('validation filenames size: ', len(val_filenames))
    print('test filenames size: ', len(test_filenames))
    print('label counts for training set: ', get_counts(train_filenames))
    print('label counts for validation set: ', get_counts(val_filenames))
    print('label counts for test set: ', get_counts(test_filenames))

    train_dataset = OASIS(train_filenames[:3])
    val_dataset = OASIS(val_filenames[:1])
    test_dataset = OASIS(test_filenames[:3])
    '''print([y for img, y in train_dataset])
    print([y for img, y in val_dataset])
    print([y for img, y in test_dataset])'''

    #print out a sample image shape
    '''image_array, label = train_dataset[4]
    print(image_array.shape)'''
    print('training dataset size: ', len(train_dataset))
    print('validation dataset size: ', len(val_dataset))
    print('test dataset size: ', len(test_dataset))
    trainset_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
    valset_loader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=4)
    testset_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=4)

    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    inception = torchvision.models.inception_v3()
    # Since imagenet has 1000 classes, we need to change our last layer according to the number of classes we have
    n_features = inception.fc.in_features
    inception.fc = nn.Linear(n_features, n_classes)

    # Freeze layers if freeze_layer is True
    for i, param in inception.named_parameters():
        if freeze_layers:
            param.requires_grad = False
        else:
            param.requires_grad = True
    if freeze_layers:
        ct = []
        for name, child in inception.named_children():
            #if name == 'fc':
            if start_freeze_layer in ct:
                for params in child.parameters():
                    params.requires_grad = True
            ct.append(name)
    # To view which layers are freezed and which layers are not freezed:
    for name, child in inception.named_children():
        for name_2, params in child.named_parameters():
            print(name_2, params.requires_grad)

    if use_parallel:
        print("[Using all the available GPUs]")
        inception = nn.DataParallel(inception, device_ids=[0, 1])

    dataloaders = {'train': trainset_loader, 'val': valset_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    optimizable_params = [param for param in inception.parameters() if param.requires_grad]
    optimizer = optimizer_type(optimizable_params, lr=0.001)
    exp_lr_scheduler = lr_scheduler_type(optimizer, step_size=7, gamma=0.1)
    # If we want to load a model with saved parameters
    if load_model_filepath is not None:
        load_saved_model(load_model_filepath, inception, optimizer)
    model_list, best_model = train_model(inception,
                             dataloaders,
                             dataset_sizes,
                             criterion,
                             optimizer,
                             exp_lr_scheduler,
                             use_cuda,
                             num_epochs)
    
    
    for model in model_list:
        predictions = evaluate_model(model, testset_loader, len(test_dataset), use_cuda)
        true_y = [y for img, y in test_dataset]
        print(classification_report(true_y, predictions))

    '''predictions = evaluate_model(best_model, testset_loader, len(test_dataset), use_cuda)
    true_y = [y for img, y in test_dataset]
    print(classification_report(true_y, predictions))'''


if __name__ == "__main__":
    run()
