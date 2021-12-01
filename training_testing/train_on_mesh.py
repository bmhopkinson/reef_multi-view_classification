import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from meshtrain_dataset import MeshTrainDataset, MeshTrainDataset_RNN
from nviewnet_v2 import nViewNet_RNN, nViewNet
from rn152 import RN152
from torch.autograd import Variable
from torch.utils.data import DataLoader
from helper import count_parameters
import itertools

"""
nViewNet
for Pytorch 0.40

Typical training occurs in two phases

1. Train Resnet152 (top and all)
    model_to_train = "ResNet152"
    lr_top = 1e-4
    lr_all = 1e-5
    epochs_top = 150
    epochs_all = 100

2. Train nViewNet (top only)
    model_to_train = "nViewNet"
    lr_top = 1e-4
    lr_all = 1e-5
    epochs_top = 150
    epochs_all = 0

    resume = True
    resume_model = "SPECIFY MODEL FILENAME"
"""


#
# nViewNet Configurations
#

model_to_train = 'nViewNet_RNN'  # choices are 'ResNet152' , 'nViewNet',  'nViewNet_RNN', 'nViewNet_resume'
model_dir = "models"

pretrained_resnet_file = "ResNet152_20211112.torch"   # need pretrained weights for nViewNet "nViewNet_ResNet152_Initial_Weights.torch"
resume_nViewNet_file = "nViewNet-8_simple_20181022.torch"  #if model_to_train == nViewNet_resume specify file name here

# Experiment Parameters
n_class = 14  # 11 for standard class set; 14 for augmented classes
n_views = 8  # set to 8 for nViewNet-8, 4 for nViewNet-4, etc


# HyperParameters
batchsize_top = 32
batchsize_all = 16
epochs_top = 50# number of epochs to train the top of the model
epochs_all = 0 # number of epochs to train the entire model - nViewNet => 0

lr_top = 1e-4 # learning rate for training the top of your model
lr_all = 1e-5 # learning rate to use when training the entire model
#momentum = 0.8  #for SGD

# Dataset infiles
#train_infile = 'train_infile_ML_215.txt'
#val_infile  = 'val_infile_ML_215.txt'
train_infile = './infiles/train_infile_20190503_small.txt'
val_infile   = './infiles/val_infile_20190503_small.txt'
#train_infile = './infiles/train_infile_20190503_noG4.txt'
#val_infile   = './infiles/val_infile_20190503_noG4.txt'
#
# Print Configs

if model_to_train == "ResNet152":
    configs = "{}_LRtop_{}_LRall_{}".format(model_to_train, lr_top, lr_all)
elif model_to_train == "nViewNet":
    configs = "{}-{}_LRtop_{}_LRall_{}".format(model_to_train, n_views, lr_top, lr_all)
elif model_to_train == "nViewNet_RNN":
    configs = "{}-RNN_LRtop_{}_LRall_{}".format(model_to_train, lr_top, lr_all)
elif model_to_train == "nViewNet_resume":
    configs = "{}-{}_LRtop_{}_LRall_{}".format(model_to_train, n_views, lr_top, lr_all)

print("Configs:", configs)

#
# Create Models Dir
#
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)

# create dir for score
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
val_scores = [] #np.zeros(epochs_top + epochs_all)

#
# GPU/CPU Setup
#
use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def collate_fn_padimg(data):
    #{'X': np_series, 'Y': ann, 'elm': elm, 'img_fnames': img_fnames, 'len': len(image_series)}
    batch_size = len(data)

    anns = torch.tensor([x['Y'] for x in data], dtype=torch.long)
    elms = torch.tensor([x['elm'] for x in data], dtype=torch.long)
    img_fnames = [x['img_fnames'] for x in data]
    img_fnames = [list(x) for x in itertools.zip_longest(*img_fnames)]

    lens = [x['len'] for x in data]
    max_len = max(lens)
    lens = torch.tensor(lens, dtype=torch.long)

    #setup image
    img_sample = data[0]['X']
    _, c, w, h = img_sample.shape
    imgs = torch.zeros(batch_size, max_len, c, w, h)  #zero-padded, merged image series
    for i, datum in enumerate(data):
        img = datum['X']
        n_elms = datum['len']
        imgs[i, 0:n_elms, :, :, :] = torch.from_numpy(img)

    return {'X': imgs, 'Y': anns, 'elm': elms, 'img_fnames': img_fnames, 'len': lens}


#
# Data Loaders and Datasets
#
def setup_dataloaders(dataset_dict, batch_size, bShuffle, num_workers):
    dataloaders = {}
    for key in dataset_dict:
        dataloaders[key] = DataLoader(dataset_dict[key], batch_size=batch_size, shuffle=bShuffle, collate_fn= collate_fn_padimg,  num_workers=num_workers)
    return dataloaders
#
# Create Model
#
def setup_model():

    if model_to_train == "ResNet152":
        pretrained_model = torchvision.models.resnet152(pretrained=True)
        resnet_bottom = torch.nn.Sequential(*list(pretrained_model.children())[:-1]) # remove last layer (fc) layer
        model = RN152(base_model=resnet_bottom, num_classes=n_class)
    elif model_to_train == "nViewNet":
    # load pretrained resnet weights and then continue to train the top
        model_loc = os.path.join(model_dir, pretrained_resnet_file)
        resnet_bottom = torch.load(model_loc)
        resnet_bottom = torch.nn.Sequential(*list(resnet_bottom.children())[:-1])
        model = nViewNet(base_model=resnet_bottom, num_classes=n_class, num_views = n_views)

    elif model_to_train == "nViewNet_RNN":
    # load pretrained resnet weights and then continue to train the top
        model_loc = os.path.join(model_dir, pretrained_resnet_file)
        resnet_bottom = torch.load(model_loc)
        resnet_bottom = torch.nn.Sequential(*list(resnet_bottom.children())[:-1])
        model = nViewNet_RNN(base_model=resnet_bottom, num_classes=n_class, num_views = n_views)

    elif model_to_train =='nViewNet_resume':
        model_loc = os.path.join(model_dir, resume_nViewNet_file)
        model = torch.load(model_loc)

    return model


log_file = open("train_on_mesh_v3_logfile.txt","w")
# mimic structre of train_model() function here: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def train(model, dataloaders, criterion, optimizer, num_epochs, scheduler = lr_scheduler, best_acc=0):
    first_pass = True
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                if scheduler:
                    scheduler.step()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_samples = 0

            for it, batch in enumerate(dataloaders[phase]):
                image_series = batch['X'].to(device)
                target = batch['Y'].to(device)

                if model_to_train == 'nViewNet_RNN':
                    data = (image_series, batch['len'].to(device))
                else:
                    data = image_series

                if first_pass:
                    print('first pass: target: {}'.format(target))
                    print('img_files: {}'.format(batch['img_fnames']))
                    first_pass = False

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase =='train'):  #track gradients for backprop in training phase
                    output = model(data)  # pass in image series
                    _, preds = torch.max(output, 1)
                    loss = criterion(output, target)  # evaluate loss

                    if phase == 'train':
                        loss.backward()  # update the gradients
                        optimizer.step()  # update model parameters based on backprop

                running_loss += loss.item() * image_series.size(0)
                running_corrects += torch.sum(preds == target.data)
                running_samples += image_series.size(0)

                if it % 10 == 0:
                    avg_loss= running_loss/(running_samples)
                    print("Epoch:", epoch, "Iteration:", it, "Average Loss:", avg_loss)  # print the running/average loss, iterat starts at 0 thus +1

            # save model if it is the best model on val set
            epoch_loss = running_loss/running_samples
            epoch_acc = running_corrects.double()/running_samples
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            log_file.write('epoch\t{}\tphase\t{}\tLoss\t{:.4f}\tAcc\t{:.4f}\n'.format(epoch, phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model, model_path)

    return best_acc


if __name__ == "__main__":
    #setup datasets


    if model_to_train == "nViewNet_RNN":
        train_data = MeshTrainDataset_RNN(train_infile,  ann_ones_based=True)
        val_data = MeshTrainDataset_RNN(val_infile, ann_ones_based=True)

    else:
        if model_to_train == "ResNet152":
            n_views = 1
        train_data = MeshTrainDataset(train_infile, nview_len=n_views, ann_ones_based=True)
        val_data = MeshTrainDataset(val_infile, nview_len=n_views, ann_ones_based=True)

    datasets = {'train' : train_data, 'val': val_data}
    bShuffle = False
    num_workers = 16
    dataloaders_top = setup_dataloaders(datasets, batchsize_top, bShuffle, num_workers)
    dataloaders_all = setup_dataloaders(datasets, batchsize_all, bShuffle, num_workers)

    #setup model
    model = setup_model()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Train on Top
    # Freeze bottom of model so we are only training the top linear layer

    for param in model.parameters():
        param.requires_grad = False

    if model_to_train == "ResNet152":
        params_to_optimize_in_top = list(model.fc.parameters())
    elif model_to_train in ["nViewNet", "nViewNet_RNN", "nViewNet_resume"]:
        params_to_optimize_in_top = list(model.collapse.parameters()) + list(model.fc.parameters())

    for param in params_to_optimize_in_top:
        param.requires_grad = True

    #optimizer_top = optim.SGD(params_to_optimize_in_top, lr=lr_top, momentum=momentum)
    optimizer_top = optim.Adam(params_to_optimize_in_top, lr = lr_top)
    #lr_scheduler_top = lr_scheduler.StepLR(optimizer_top, step_size=20, gamma=0.8)
    lr_scheduler_top = None
    print("Training Top:", count_parameters(model), "Parameters")

    b_acc = train(model, dataloaders_top, criterion, optimizer_top, epochs_top, scheduler = lr_scheduler_top, best_acc=0)
    print('Finished training top, best acc {:.4f}'.format(b_acc))
    # Set all parameters to train (require gradient)
    for param in model.parameters():
        param.requires_grad = True
    print("Training All:", count_parameters(model), "Parameters")

    # Optimizer for Entire Network
    #optimizer_all = optim.SGD(model.parameters(), lr=lr_all, momentum=momentum)
    optimizer_all = optim.Adam(model.parameters(), lr = lr_all)
    #lr_scheduler_all = lr_scheduler.StepLR(optimizer_all, step_size=10, gamma=0.8)
    lr_scheduler_all = None
    # Train on All
    b_acc = train(model, dataloaders_all, criterion, optimizer_all, epochs_all, scheduler = lr_scheduler_all, best_acc=b_acc)
    print('Finished training all, best acc {:.4f}'.format(b_acc))
