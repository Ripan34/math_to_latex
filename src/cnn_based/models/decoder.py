import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, encoder_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.lstm = nn.LSTM(embed_dim + encoder_dim, hidden_dim, batch_first=True)

        self.attn_hidden = nn.Linear(hidden_dim, encoder_dim)
        self.attn_enc = nn.Linear(encoder_dim, encoder_dim)
        self.attn_score = nn.Linear(encoder_dim, 1)

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def compute_attention(self, features, hidden):
        hidden_proj = self.attn_hidden(hidden).unsqueeze(1)

        enc_proj = self.attn_enc(features)

        scores = self.attn_score(torch.tanh(enc_proj + hidden_proj))
        scores = scores.squeeze(-1)

        attn_weights = F.softmax(scores, dim=1)

        context = torch.bmm(attn_weights.unsqueeze(1), features)
        return context

    def forward(self, features, captions):
        embeddings = self.embedding(captions)

        batch_size = captions.size(0)

        hidden = None
        outputs = []

        for t in range(captions.size(1)):
            token_embed = embeddings[:, t:t+1, :]

            if hidden is None:
                h = torch.zeros(batch_size, self.lstm.hidden_size).to(features.device)
            else:
                h = hidden[0].squeeze(0)

            # context
            context = self.compute_attention(features, h)

            # LSTM input = token embedding + context
            lstm_input = torch.cat([token_embed, context], dim=-1)

            # Run LSTM
            lstm_out, hidden = self.lstm(lstm_input, hidden)

            out = self.fc(lstm_out)
            outputs.append(out)

        outputs = torch.cat(outputs, dim=1)
        return outputs
