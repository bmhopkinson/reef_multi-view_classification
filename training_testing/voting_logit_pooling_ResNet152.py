#predict on test data set and save preds/lables for Confusion Matrix

import numpy as np
import sys
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from meshtrain_dataset import MeshTrainDataset
from rn152 import RN152

# Dataset, model infiles, outfilesD
test_infile  = './infiles/test_infile_20190503.txt'
outfile_logit = "logit_pooling_test_preds_labels.txt"
outfile_voting = "voting_test_preds_labels.txt"
model_path = './models/ResNet152_trained_FL_201905.torch'
n_classes = 11
#
# Data Loaders and Datasets
#
batch_size = 1  # I don't expect to use this script much so can only handle batch size 1
nview_len = 16

test_data = MeshTrainDataset(test_infile, nview_len = nview_len, ann_ones_based = True )
data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8 )
n_elms = len(test_data)

resnet = torch.load(model_path)
resnet.eval()  # this preps your dropout and batchnorm layers for validation
device = torch.device("cuda")  #model is set for gpu
softmax = torch.nn.Softmax(1)

#setup vectors to hold predictions

y_pred_logit = []
y_pred_vote = []
y_label = []

#loop through samples and make a prediction for each meshI
for it, sample in enumerate(data_loader):
    with torch.no_grad():
        image_series = sample['X'].to(device)  #acquire image data
        logit_pool = np.zeros((nview_len, n_classes))
        votes = np.zeros(n_classes, dtype=int)

        for i in range(0, nview_len):
            one_im = image_series[..., i].float()
            one_im = one_im.unsqueeze(4)  #this is dumb but relates to old structure of andrew's slight mods in rn152.py
            output = resnet(one_im).cpu()  # pass individual image through resnet152
            logits = output.data  

            this_vote = output.data.max(1, keepdim=True)[1].numpy().astype(int)	 
            votes[this_vote]  =  votes[this_vote] + 1
            prob = softmax(logits)
            logit_pool[i,:] = prob

            y_pred_logit.append(np.argmax(np.mean(logit_pool,axis = 0)))
            y_pred_vote.append(np.argmax(votes))
            y_label.append(sample['Y'].numpy().astype(int))

    if it % 10 == 0:
        print(it*batch_size)

with open(outfile_logit, 'w') as outport:
    for l, p in zip(y_label, y_pred_logit):
        outport.write('%d %d\n' % (l, p) )

with open(outfile_voting, 'w') as outport:
    for l, p in zip(y_label, y_pred_vote):
        outport.write('%d %d\n' % (l, p) )

