import torch
import torch.nn as nn
from transformers import ViTModel

class ViTEncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.vit.config.hidden_size, 512)

    def forward(self, x):
        x = x.repeat(1,3,1,1)  # ViT expects 3 channels
        out = self.vit(pixel_values=x).last_hidden_state
        return self.proj(out)