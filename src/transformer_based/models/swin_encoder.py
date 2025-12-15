import torch.nn as nn
from transformers import SwinModel

class SwinEncoder(nn.Module):
    def __init__(self, model_name="microsoft/swin-tiny-patch4-window7-224"):
        super().__init__()
        self.swin = SwinModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.swin.config.hidden_size, 512)

    def forward(self, x):
        # Swin expects 3-channel RGB images → replicate grayscale 3 times
        x = x.repeat(1, 3, 1, 1)

        out = self.swin(pixel_values=x).last_hidden_state
        out = self.proj(out)
        return out