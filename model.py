# Final model used on CodaLab

import torch
import torch.nn as nn
import torchvision.models.segmentation as models
from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet50_Weights

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        self.model.classifier[4] = nn.Conv2d(256, 19, kernel_size=1)

    def forward(self, x):
        return self.model(x)