"""
Fully Convolutional Network (FCN) for semantic segmentation.
"""

import torch
import torchvision

class FCN(torch.nn.Module):

    def __init__(self, n_channels, num_classes):
        super(FCN, self).__init__()

        self.model                = torchvision.models.segmentation.fcn_resnet50(num_classes=num_classes)
        self.model.backbone.conv1 = torch.nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        return self.model(x)['out']
    
    
    def load_weights(self, path):
        """Load model weights from a file.

        Args:
        ----------------
            path (str): Path to the file containing the weights.
        """
        from collections import OrderedDict
        state_dict = torch.load(path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("model.", "")] = v
        self.model.load_state_dict(new_state_dict)