import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=4, num_heads=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, memory):
        tgt_emb = self.embed(tgt)
        out = self.decoder(tgt_emb, memory)
        return self.fc(out)