#!/usr/bin/env python3

import numpy as np
import cv2
import torch
import sys
import math
import os

import meshview_dataset_helpers as helper
from nviewnet_v2 import nViewNet
from meshview_dataset import MeshViewDataset
from torch.utils.data import DataLoader


model_file = "./models/nViewNet-8_sample.torch"
batch_size = 10
max_loads = 500000  #maximum number of times the dataloader is called due to untraced memory leak

#   FILENAMES of data to load

baseDir = "./sample_data/"
imgList = "sample_data_images.txt"
visFile = "sample_data_faceVisSparse_Alt.mat"
outfile = "sample_data_nViewNet_testing.txt"

cpu = torch.device("cpu")
gpu = torch.device("cuda")

#load mesh to image visibility information and process
imCoord_x, imCoord_y = helper.load_imCoord_alt(baseDir + visFile)
pts =  helper.create_pt_datastruct_alt(imCoord_x, imCoord_y)
meshElmViews = helper.group_pts_by_meshID(pts, 'List')
n_elms = len(meshElmViews)
n_batches = math.ceil(n_elms/batch_size)
max_batches_per_loop = math.floor(max_loads/batch_size)
n_dsets = math.ceil(n_batches/max_batches_per_loop)

#n_dsets = math.ceil(n_elms/max_loads)

#load cnn
nviewnet = torch.load(model_file)
nviewnet.eval()  # this preps your dropout and batchnorm layersfrom torch.autograd import Variable for validation


#setup output vectors
y_pred = np.zeros([n_elms,1], dtype= int)  #slightly odd array shape required to match output format
y_id   = np.zeros([n_elms], dtype= int)
i_s = 0  		#batch start index
i_e = batch_size	#batch end index

#loop through samples and make a prediction for each meshID
with torch.no_grad():
    for i in range(0,n_dsets):  #in training, this loop occurs every epoch perhaps explaining why training doesn't run out of memory
        idx_start = i*max_batches_per_loop*batch_size  #index offset
        idx_end  = min(n_elms, idx_start + max_batches_per_loop*batch_size)

        trial_data = MeshViewDataset(baseDir, meshElmViews, imgList, idx_start, idx_end)
        data_loader = DataLoader(trial_data, batch_size=batch_size, num_workers=16)

        for it, sample in enumerate(data_loader):
            output = nviewnet(sample['views'].to(gpu)).to(cpu)		     #pass through nviewnet
            temp = output.data.max(1, keepdim=True)      # for debugging
            pred = output.data.max(1, keepdim=True)[1]	 # predicted class is one with max logit score

            i_s = idx_start + it*batch_size
            i_e = i_s + batch_size
            if i_e > idx_end:
               i_e = idx_end

            y_pred[i_s:i_e] = pred.numpy().astype(int)
            y_id[i_s:i_e]   = sample['meshID'].numpy().astype(int)

            if it % 100 == 0:
               print(it*batch_size)

        y_temp = np.empty([n_elms,2])
        y_temp[:,0] = y_id
        y_temp[:,1] = y_pred[:,0]
        np.savetxt('intermediate_pred.txt',y_temp, fmt = '%d')


#save results
y_all = np.empty([n_elms,2])
y_all[:,0] = y_id
y_all[:,1] = y_pred[:,0]
np.savetxt(outfile,y_all, fmt = '%d')
