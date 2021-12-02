#predict on test data set and save preds/lables for Confusion Matrix

import numpy as np
import sys
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

#sys.path.append('~/Documents/nViewNet/common')
from nviewnet import nViewNet
from meshtrain_dataset import MeshTrainDataset


# Model, Dataset infiles, outfiles
model_file = "./models/nViewNet-8_noG4_wResNetnoG4.torch"
test_infile  = './infiles/test_infile_G4.txt'
outfile = "nViewNet-8_modelnoG4_wResNetnoG4_G4_preds.txt"
n_rep = 1  #number of times to repeat prediction (randomness in which views are chosen)

#
# Data Loaders and Datasets
#
batch_size = 16
nview_len = 8

device = torch.device("cuda")  #model is set for gpu

test_data = MeshTrainDataset(test_infile, nview_len = nview_len, ann_ones_based = True )
data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8 )
n_elms = len(test_data)

nviewnet = torch.load(model_file)
nviewnet.eval()  # this preps your dropout and batchnorm layers for validation

##setup vectors to hold predictions
y_pred  = np.zeros([n_elms,n_rep] , dtype= int)
y_elm   = np.zeros([n_elms,n_rep] , dtype= int)
y_label = np.zeros([n_elms,n_rep] , dtype= int)

#loop through samples and make a prediction for each meshID
for i in range(n_rep):
    for it, sample in enumerate(data_loader):
        with torch.no_grad():
            image_series = sample['X'].to(device)  #acquire image data
            output = nviewnet(image_series).cpu()		     #pass through nviewnet
            temp = output.data.max(1, keepdim=True)      # for debugging
            pred = output.data.max(1, keepdim=True)[1]	 # predicted class is one with max logit score

            i_s = it*batch_size
            i_e = (it+1)*batch_size
            if i_e > n_elms:
                i_e = n_elms

            y_pred[i_s:i_e,i]  = np.squeeze(pred.numpy().astype(int))
            y_elm[i_s:i_e,i]   = sample['elm'].numpy().astype(int)
            y_label[i_s:i_e,i] = sample['Y'].numpy().astype(int)


            if it % 10 == 0:
                print('rep {}, it {}'.format(i, it))

y_all = np.empty([n_elms,n_rep*3])
y_all[:,0:n_rep]         = y_elm
y_all[:,n_rep:2*n_rep]   = y_label
y_all[:,2*n_rep:3*n_rep] = y_pred
np.savetxt(outfile,y_all, fmt = '%d')
