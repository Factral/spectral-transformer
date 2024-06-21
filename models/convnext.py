"""
UperNet framework for semantic segmentation, leveraging a ConvNeXt backbone.
Code adapted from: https://huggingface.co/openmmlab/upernet-convnext-small
"""

import torch
from transformers import ConvNextConfig, ConvNextModel
from transformers import UperNetConfig, UperNetForSemanticSegmentation

class ConvNext(torch.nn.Module):
    def __init__(self, n_channels, n_classes):

        super(ConvNext, self).__init__()

        backbone_config = ConvNextConfig(num_channels=n_channels, out_features=["stage1", "stage2", "stage3", "stage4"])
        config = UperNetConfig(backbone_config=backbone_config, num_labels=n_classes)

        self.model = UperNetForSemanticSegmentation(config)

    def forward(self, x):

        out  = self.model(x, return_dict=True)
        return out.logits