import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, 1)

    def forward(self, query, keys):
        q = self.Wq(query).unsqueeze(1)
        k = self.Wk(keys)
        scores = self.v(torch.tanh(q + k)).squeeze(-1)
        attn = F.softmax(scores, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), keys).squeeze(1)
        return context, attn


class DotProductAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scale = d_model ** -0.5

    def forward(self, query, keys):
        scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)
        attn = F.softmax(scores * self.scale, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), keys).squeeze(1)
        return context, attn