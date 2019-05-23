import os
import sys

import numpy as np
import pandas as pd
import random
import torch
from skimage import io
from torch.utils.data import Dataset
import cv2

import meshview_dataset_helpers as helper

NVIEW_LEN = 8  #number of views used in nviewnet cnn
RGB_MEANS = np.array([91.49748071, 166.89248008, 123.99212699]) / 255.  # mean of three channels in the order of RGB


class MeshViewDataset(Dataset):
    def __init__(self, base_dir, meshElmViews, img_filenames, idx_start, idx_stop):
        self.means = RGB_MEANS
        self.base_dir = base_dir

        #load list of raw image filenames
        self.img_filenames = helper.load_imgfiles(base_dir + img_filenames, base_dir)

        self.meshElmViews = meshElmViews[idx_start:idx_stop]

    def __len__(self):
        return len(self.meshElmViews)

    def __getitem__(self, idx):
        image_series = []

        elm   = self.meshElmViews[idx][0]
        views = self.meshElmViews[idx][1]

        nViews = len(views)

        views_sel = []  #select NVIEW_LEN views from all available for this element
        if nViews == NVIEW_LEN:
            views_sel = views
        if nViews <  NVIEW_LEN:
            views_sel = views
            for i in range(0,NVIEW_LEN - nViews):
                views_sel.append(views_sel[i])  #not perfect, but should work
        if nViews >  NVIEW_LEN:
            views_sel = random.sample(views, NVIEW_LEN);


        for this_view in views_sel:
            imfile = self.img_filenames[this_view[0]]
        #    fullPath = self.base_dir + imfile[1:]
        #    print(fullPath)
            x = this_view[1]
            y = this_view[2]
            im = cv2.imread(imfile)
            patch = helper.extract_patch(im, 100, x, y)
            patch = self.subtract_mean(patch)
            image_series.append(patch)

	#restructure image series
        np_series = np.zeros(shape=(3, 200, 200, len(image_series)))
        for idx2, im in enumerate(image_series):
            c, w, h = im.shape
            np_series[:c, :w, :h, idx2] = im

      #  return {'meshID': elm, 'views': image_series}  #for debugging
        return {'meshID': elm, 'views': np_series}
    # TRANSPOSE AND SUBTRACT MEAN
    def subtract_mean(self, img):
        img = np.transpose(img, (2, 0, 1)) / 255.  # rearrange array to channels, width, height and normalize
        img[0] -= self.means[0]
        img[1] -= self.means[1]
        img[2] -= self.means[2]
        return img
