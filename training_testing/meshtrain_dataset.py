import os

import csv
import numpy as np
import pandas as pd
import random
import torch
from skimage import io
from torch.utils.data import Dataset
import cv2

import meshview_dataset_helpers as helper


class MeshTrainDataset(Dataset):
    def __init__(self, infile, nview_len, ann_ones_based):
        self.means = np.array([91.49748071, 166.89248008, 123.99212699]) / 255.  # mean of three channels in the order of RGB
        self.nview_len = nview_len  #number of views returned; should match number used in nViewNet
        self.ann_ones_based = ann_ones_based
        meshElmViews = []
        imgFiles = []
        with open(infile,'r') as f:
            reader=csv.reader(f,delimiter='\t')

            for n, base_dir, anns, vis, img in reader:
                   print(vis)
                   imCoord_x,imCoord_y = helper.load_imCoord_alt(base_dir + vis)
                   meshID_anns = helper.load_anns(base_dir + anns)  # THESE ANNOTATIONS LIKELY USE ONES BASED INDEXING FOR MESHIDS - convert in this function
                   offset = len(imgFiles)
                   #print(offset)
                   theseViews = helper.create_train_segment_alt(n, imCoord_x,imCoord_y ,meshID_anns, offset)
                   meshElmViews.extend(theseViews)
                   imgFiles.extend(helper.load_imgfiles(base_dir + img, base_dir))


        self.meshElmViews = meshElmViews
        self.img_filenames = imgFiles



    def __len__(self):
        return len(self.meshElmViews)

    def __getitem__(self, idx):
        image_series = []

        elm   = self.meshElmViews[idx][1]
        ann   = self.meshElmViews[idx][2]
        if self.ann_ones_based:
            ann = ann - 1
      #  print(ann)
        views = self.meshElmViews[idx][3]

        nViews = len(views)

        views_sel = []
        if nViews == self.nview_len:
            views_sel = views
        if nViews <  self.nview_len:
            views_sel = views
            for i in range(0,self.nview_len - nViews):
                views_sel.append(views_sel[i])  #not perfect, but should work
        if nViews >  self.nview_len:
            views_sel = random.sample(views, self.nview_len);


        for this_view in views_sel:
            fullPath = self.img_filenames[this_view[0]]
         #   fullPath = self.root_dir + imfile[1:]
         #   print(fullPath)
            x = this_view[1]
            y = this_view[2]
            im = cv2.imread(fullPath)
            patch = helper.extract_patch(im, 100, x, y)
            patch = self.subtract_mean(patch)
            image_series.append(patch)

	#restructure image series
        np_series = np.zeros(shape=(3, 200, 200, len(image_series)))
        for idx2, im in enumerate(image_series):
            c, w, h = im.shape
            np_series[:c, :w, :h, idx2] = im

     #    return {'meshID': elm, 'views': image_series, 'ann' : ann}  #for debugging
        return {'X': np_series, 'Y': ann, 'elm': elm}
    # TRANSPOSE AND SUBTRACT MEAN
    def subtract_mean(self, img):
        img = np.transpose(img, (2, 0, 1)) / 255.  # rearrange array to channels, width, height and normalize
        img[0] -= self.means[0]
        img[1] -= self.means[1]
        img[2] -= self.means[2]
        return img
