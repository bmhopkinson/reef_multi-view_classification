import torch.nn as nn


class RN152(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()  # don't need to pass class in python 3 so NOT super(TwinNet, self).__init()
        self.base = base_model
        self.n_classes = num_classes
        self.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)

    def forward(self, image_series, ):
        # we run our images series through resnet and have them in a 5d tensor (batch, channels, width, height, series)
        pooled = self.base(image_series[..., 0].float())

        pooled = pooled.view(pooled.size(0), -1)  # this is a flatten layer so it passes correctly to our linear layer; retain batch size (dim 0), flatten the rest
        pooled = self.fc(pooled)
        return pooled
