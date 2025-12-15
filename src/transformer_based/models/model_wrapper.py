import torch
import torch.nn as nn
import torch.nn.functional as F

class HMERModel(nn.Module):
    def __init__(self, encoder, decoder, attention):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention

    def forward(self, images, tgt_tokens):
        enc = self.encoder(images)         # [B, S, 512]

        # last decoder hidden state approximated as autoregressive teacher-forcing
        query = self.decoder.embed(tgt_tokens)[:, -1]  

        context, attn = self.attention(query, enc)

        logits = self.decoder(tgt_tokens, enc)

        return logits, attn