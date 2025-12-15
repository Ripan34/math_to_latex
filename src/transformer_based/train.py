import os
import time
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import MathWritingDataset
from data.tokenizer import LatexTokenizer
from data.collate import collate_fn

from models.vit_encoder import ViTEncoder
from models.swin_encoder import SwinEncoder
from models.attention import AdditiveAttention, DotProductAttention
from models.transformer_decoder import TransformerDecoder
from models.model_wrapper import HMERModel


# -----------------------------------------------------------
# Device selection
# -----------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple Silicon MPS backend")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


# -----------------------------------------------------------
# Load YAML configuration
# -----------------------------------------------------------
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------
# Training script
# -----------------------------------------------------------
def main(config_path):
    device = get_device()
    cfg = load_config(config_path)

    # -------------------------------------------------------
    # Tokenizer + dataset
    # -------------------------------------------------------
    tokenizer = LatexTokenizer()

    train_ds = MathWritingDataset(
        split="train",
        tokenizer=tokenizer,
        img_size=cfg["data"]["image_size"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=False,   # MPS does not support pinned memory
    )

    # -------------------------------------------------------
    # Encoder selection (ViT or Swin)
    # -------------------------------------------------------
    enc_type = cfg["model"]["encoder_type"]
    enc_name = cfg["model"]["encoder_name"]

    if enc_type == "vit":
        encoder = ViTEncoder(enc_name)
    elif enc_type == "swin":
        encoder = SwinEncoder(enc_name)
    else:
        raise ValueError(f"Unknown encoder_type '{enc_type}'")

    # -------------------------------------------------------
    # Attention module (additive or dot-product)
    # -------------------------------------------------------
    if cfg["model"]["attention_type"] == "additive":
        attention = AdditiveAttention(cfg["model"]["d_model"])
    else:
        attention = DotProductAttention(cfg["model"]["d_model"])

    # -------------------------------------------------------
    # Decoder
    # -------------------------------------------------------
    vocab_size = len(tokenizer.id2tok) + 200
    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=cfg["model"]["d_model"],
        num_layers=cfg["model"]["num_layers"],
        num_heads=cfg["model"]["num_heads"],
    )

    # -------------------------------------------------------
    # Build full model
    # -------------------------------------------------------
    model = HMERModel(encoder, decoder, attention).to(device)

    # -------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------
    lr = float(cfg["training"]["lr"])
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    num_epochs = cfg["training"]["num_epochs"]

    # -------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------
    for epoch in range(num_epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", total=len(train_loader))

        for batch in progress:
            images = batch["image"].to(device)
            tokens = batch["tokens"].to(device)

            # forward pass
            logits, _ = model(images, tokens[:, :-1])

            # compute loss
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tokens[:, 1:].reshape(-1),
                ignore_index=0,
            )

            optim.zero_grad()
            loss.backward()
            optim.step()

            progress.set_postfix({"loss": loss.item()})

        # ---------------------------------------------------
        # Save checkpoint WITH metadata
        # ---------------------------------------------------
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        metadata = {
            "encoder_type": enc_type,
            "encoder_name": enc_name,
            "attention_type": cfg["model"]["attention_type"],
            "d_model": cfg["model"]["d_model"],
            "num_layers": cfg["model"]["num_layers"],
            "num_heads": cfg["model"]["num_heads"],
            "vocab_size": vocab_size,
        }

        save_obj = {
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        }

        ckpt_name = f"checkpoint_epoch_{epoch+1}_{timestamp}.pt"
        torch.save(save_obj, ckpt_name)

        print(f"\nSaved checkpoint: {ckpt_name}\n")


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    main(args.config)