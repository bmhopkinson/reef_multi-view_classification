import os
import csv
import numpy as np
import random
import itertools
import torch
from torch.utils.data import Dataset
import cv2

import meshview_dataset_helpers as helper

class MeshTrainDataset(Dataset):
    def __init__(self, infile, ann_ones_based, nview_len=8, RNN=False):
        self.means = np.array(
            [91.49748071, 166.89248008, 123.99212699]) / 255.  # mean of three channels in the order of RGB
        self.ann_ones_based = ann_ones_based
        self.nview_len = nview_len  # temp for development
        self.RNN = RNN

        meshElmViews = []
        imgFiles = []
        with open(infile, 'r') as f:
            reader = csv.reader(f, delimiter='\t')

            for n, base_dir, anns, vis, img in reader:
                print(vis)
                imCoord_x, imCoord_y = helper.load_imCoord_alt(base_dir + vis)
                meshID_anns = helper.load_anns(
                    base_dir + anns)  # THESE ANNOTATIONS LIKELY USE ONES BASED INDEXING FOR MESHIDS - convert in this function
                offset = len(imgFiles)
                # print(offset)
                theseViews = helper.create_train_segment_alt(n, imCoord_x, imCoord_y, meshID_anns, offset)
                meshElmViews.extend(theseViews)
                imgFiles.extend(helper.load_imgfiles(base_dir + img, base_dir))

        self.meshElmViews = meshElmViews
        self.img_filenames = imgFiles

    def __len__(self):
        return len(self.meshElmViews)

    def __getitem__(self, idx):
        image_series = []

        elm = self.meshElmViews[idx][1]
        ann = self.meshElmViews[idx][2]
        if self.ann_ones_based:
            ann = ann - 1
        #  print(ann)
        views = self.meshElmViews[idx][3]


        if not self.RNN:
            views = self.sample_views(views)

        img_fnames = []
        for this_view in views:
            fullPath = self.img_filenames[this_view[0]]
            file_parts = os.path.split(fullPath)
            img_fnames.append(file_parts[1])
            x = this_view[1]
            y = this_view[2]
            im = cv2.imread(fullPath)
            patch = helper.extract_patch(im, 100, x, y)
            patch = self.subtract_mean(patch)
            image_series.append(patch)

        # restructure image series
        np_series = np.zeros(shape=(len(image_series), 3, 200, 200))
        for idx2, im in enumerate(image_series):
            c, w, h = im.shape
            np_series[idx2, :c, :w, :h] = im

        return {'X': np_series, 'Y': ann, 'elm': elm, 'img_fnames': img_fnames, 'len': len(image_series)}

    # TRANSPOSE AND SUBTRACT MEAN
    def subtract_mean(self, img):
        img = np.transpose(img, (2, 0, 1)) / 255.  # rearrange array to channels, width, height and normalize
        img[0] -= self.means[0]
        img[1] -= self.means[1]
        img[2] -= self.means[2]
        return img

    def sample_views(self, views):
        nViews = len(views)
        if nViews == self.nview_len:
            views_sel = views
        if nViews < self.nview_len:
            views_sel = views
            for i in range(0, self.nview_len - nViews):
                views_sel.append(views_sel[i])  # not perfect, but should work
        else:  # nViews > self.nview_len
            views_sel = random.sample(views, self.nview_len)

        return views_sel


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