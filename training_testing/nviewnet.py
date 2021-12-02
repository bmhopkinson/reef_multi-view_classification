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
        for i in range(0, image_series.shape[1]):
            img_features = self.base(image_series[:, i, ...].float())  # pass through base cnn feature extractor (ResNet152)
            if i == 0:
                img_features_nviews = torch.squeeze(img_features)
            else:
                img_features_nviews = torch.cat([img_features_nviews, torch.squeeze(img_features)],
                                        dim=1)  # concat in 2nd dim - channels dimension
        pooled = self.collapse(img_features_nviews)
        pooled = self.fc(pooled)
        return pooled
        
        
class nViewNet_RNN(nn.Module):
    def __init__(self, base_model, num_classes, num_views):
        super().__init__()  
        self.base = base_model.eval()
        self.n_classes = num_classes
        self.hidden_size = 512
        self.collapse = nn.LSTM(input_size=2048, hidden_size=self.hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=num_classes, bias=True)
        self.num_views = num_views
        self.first_pass = True

    def forward(self, data):
        # image series processed through base CNN as a feature extractor and then passed through RNN
        # images arrive in a 5d tensor (batch, sequence, channels, width, height) with zero padding so sequence lengths are equal

        image_series = data[0]
        lens = data[1]
        cbatch_size = image_series.shape[0]  #size of the current batch

        image_series_packed = torch.nn.utils.rnn.pack_padded_sequence(image_series, lens.to(torch.device("cpu")),
                                                                      batch_first=True, enforce_sorted=False)
        self.base.eval()  #base model is being used as feature extractor. calling eval() enables running means for batchnorm layers - this is critical for efficient training
        img_features = self.base(image_series_packed.data)
        img_features_packed =torch.nn.utils.rnn.PackedSequence(torch.squeeze(img_features), batch_sizes=image_series_packed.batch_sizes,
                                                                  sorted_indices=image_series_packed.sorted_indices,
                                                                  unsorted_indices=image_series_packed.unsorted_indices)  # sequences get sorted by length - longest at top of batch,so need to pass sorting indices
        if img_features.is_cuda:
            device = img_features.get_device()
        else:
            device = torch.device('cpu')

        hidden_init = (torch.randn(1, cbatch_size, self.hidden_size).to(device), torch.randn(1, cbatch_size, self.hidden_size).to(device))
        out, hidden = self.collapse(img_features_packed, hidden_init)
        pooled = self.fc(hidden[0].squeeze(dim=0))

        return pooled
