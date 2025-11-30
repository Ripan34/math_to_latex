import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights
import ssl
import certifi

class DenseNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            # SSL verification
            model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        except ssl.SSLError:
            ssl._create_default_https_context = ssl._create_unverified_context
            model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        self.features = model.features
        
        self.output_dim = model.features[-1].num_features

    def forward(self, x):
        features = self.features(x)
        batch_size = features.size(0)
        num_channels = features.size(1)
        height = features.size(2)
        width = features.size(3)
        
        features = features.view(batch_size, num_channels, height * width)
        features = features.permute(0, 2, 1)  # (B, H*W, C)
        
        return features