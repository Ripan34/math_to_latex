import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import ssl

class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        except ssl.SSLError:
            ssl._create_default_https_context = ssl._create_unverified_context
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        
        self.output_dim = 2048

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        batch_size = x.size(0)
        num_channels = x.size(1)
        height = x.size(2)
        width = x.size(3)
        
       
        x = x.view(batch_size, num_channels, height * width)
        x = x.permute(0, 2, 1)
        
        return x
