import torch

def pad_sequence(seq_list, pad_value=0):
    max_len = max(len(seq) for seq in seq_list)
    padded = torch.full((len(seq_list), max_len), pad_value, dtype=torch.long)
    for i, seq in enumerate(seq_list):
        padded[i, :len(seq)] = seq
    return padded


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)

    token_lists = [b["tokens"] for b in batch]
    tokens = pad_sequence(token_lists, pad_value=0)

    return {
        "image": images,
        "tokens": tokens
    }