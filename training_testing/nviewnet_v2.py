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
        
        
class nViewNet_RNN(nn.Module):
    def __init__(self, base_model, num_classes, num_views):
        super().__init__()  
        self.base = base_model.eval()
        self.n_classes = num_classes
        hidden_size = 512
        self.collapse = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=num_classes, bias=True)
        self.num_views = num_views
        self.first_pass = True

    def forward(self, image_series):
        # we run our images series through resnet and have them in a 5d tensor (batch, channels, width, height, series)
        base_images = [] # each image is (batch,channels,width,height)
        cbatch_size = image_series.shape[0]  #size of the current batch

        self.base.eval()  #running means for batchnorm layers in non-trained base model - this is critical for efficient training
        for j in range(0, cbatch_size):  #process each sequence separately (preparing for time when they may have different lengths)
            features_sample = []
            img_series_sample = image_series[j].float()
            n_imgs = img_series_sample.shape[3]
            img_series_sample = torch.unsqueeze(img_series_sample, dim=0)
            img_series_sample_t = torch.transpose(img_series_sample, 0, 4)   #swap batch and sequence dimensions - all images in a sequence are fed into base as a pseudo-batch
            img_series_sample_ts = torch.squeeze(img_series_sample_t)

            out = self.base(img_series_sample_ts)

            if j == 0:
                base_images = out.reshape(1, n_imgs, -1)
            else:
                base_images = torch.cat([base_images, out.reshape(1, n_imgs, -1)], dim=0)

        if base_images.is_cuda:
            device = base_images.get_device()
        else:
            device = torch.device('cpu')

        hidden_init = (torch.randn(1, cbatch_size, 512).to(device), torch.randn(1, cbatch_size, 512).to(device))
        out, hidden = self.collapse(base_images, hidden_init)
        pooled = self.fc(hidden[0].squeeze(dim=0))

        return pooled
