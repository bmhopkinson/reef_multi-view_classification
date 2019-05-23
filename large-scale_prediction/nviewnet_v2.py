import torch
import torch.nn as nn


class nViewNet(nn.Module):
    def __init__(self, base_model, num_classes, num_views):
        super().__init__()  # don't need to pass class in python 3 so NOT super(TwinNet, self).__init()
        self.base = base_model
        self.n_classes = num_classes
        self.collapse = nn.Sequential(
            nn.Linear(in_features=2048 * num_views, out_features=2048, bias=True),
            nn.ReLU()
        )
        self.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        self.num_views = num_views

    def forward(self, image_series):
        # we run our images series through resnet and have them in a 5d tensor (batch, channels, width, height, series)
        base_images = [] # each image is (batch,channels,width,height)
        cbatch_size = image_series.shape[0]  #size of the current batch
        for i in range(0, image_series.shape[4]):
            im = self.base(image_series[..., i].float())  # pass through base cnn feature extractor (ResNet152)
        #    print('output of resnet: batch %d, channels %d, width %d, height %d' % (im.shape[0], im.shape[1], im.shape[2], im.shape[3]))
            if i == 0:
                base_images = im.reshape(cbatch_size,-1)  #dimensions 3-5 (width, height, series) are all length 1
            else:
                base_images = torch.cat([base_images, im.reshape(cbatch_size,-1)], dim=1)  # concat in 2nd dim - channels dimension
        pooled = self.collapse(base_images)
        pooled = self.fc(pooled)
        return pooled
