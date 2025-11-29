import torch
import torch.nn as nn
from torchvision.models import densenet121

class DenseNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = densenet121(pretrained=True)

        self.features = model.features

    def forward(self, x):
        return self.features(x)