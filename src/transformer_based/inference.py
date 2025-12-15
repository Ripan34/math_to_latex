import os
import glob
import torch
from PIL import Image
import torchvision.transforms as T

from data.tokenizer import LatexTokenizer
from models.vit_encoder import ViTEncoder
from models.swin_encoder import SwinEncoder
from models.attention import AdditiveAttention, DotProductAttention
from models.transformer_decoder import TransformerDecoder
from models.model_wrapper import HMERModel
from utils.visualization import visualize_attention


# -----------------------------------------------------------
# Device selection: prefer MPS on Apple Silicon
# -----------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple Silicon MPS backend")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


# -----------------------------------------------------------
# Basic preprocessing for input images
# -----------------------------------------------------------
def preprocess_image(path):
    img = Image.open(path).convert("L")
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    return transform(img).unsqueeze(0), img


# -----------------------------------------------------------
# Greedy autoregressive decoding
# -----------------------------------------------------------
def generate_sequence(model, tokenizer, img_tensor, device, max_len=120):
    model.eval()
    img_tensor = img_tensor.to(device)

    tokens = [tokenizer.tok2id[tokenizer.bos]]

    for _ in range(max_len):
        t = torch.tensor(tokens).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, attn = model(img_tensor, t)

        next_token = logits[0, -1].argmax().item()
        tokens.append(next_token)

        if next_token == tokenizer.tok2id[tokenizer.eos]:
            break

    decoded = [
        tokenizer.id2tok[x] if x < len(tokenizer.id2tok) else "<unk>"
        for x in tokens
    ]

    return decoded, attn


# -----------------------------------------------------------
# Main inference routine
# -----------------------------------------------------------
def main():
    device = get_device()
    tokenizer = LatexTokenizer()

    # -------------------------------------------------------
    # Find the most recent checkpoint
    # -------------------------------------------------------
    ckpts = glob.glob("checkpoint_epoch_*.pt")
    if len(ckpts) == 0:
        raise FileNotFoundError("No checkpoint files found. Train the model first.")

    ckpts.sort(key=os.path.getmtime)
    ckpt_path = ckpts[-1]

    print("Loading checkpoint:", ckpt_path)

    # -------------------------------------------------------
    # Load checkpoint + metadata
    # -------------------------------------------------------
    obj = torch.load(ckpt_path, map_location=device)

    if "metadata" not in obj:
        raise RuntimeError("Checkpoint missing metadata. Retrain with metadata saving enabled.")

    metadata = obj["metadata"]
    state_dict = obj["model_state_dict"]

    print("\nDetected model configuration:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")

    # -------------------------------------------------------
    # Reconstruct encoder according to metadata
    # -------------------------------------------------------
    enc_type = metadata["encoder_type"]
    enc_name = metadata["encoder_name"]

    if enc_type == "vit":
        encoder = ViTEncoder(enc_name)
    elif enc_type == "swin":
        encoder = SwinEncoder(enc_name)
    else:
        raise ValueError(f"Unknown encoder_type '{enc_type}' in checkpoint metadata.")

    # -------------------------------------------------------
    # Reconstruct attention module
    # -------------------------------------------------------
    att_type = metadata["attention_type"]

    if att_type == "additive":
        attention = AdditiveAttention(metadata["d_model"])
    else:
        attention = DotProductAttention(metadata["d_model"])

    # -------------------------------------------------------
    # Build decoder
    # -------------------------------------------------------
    decoder = TransformerDecoder(
        vocab_size=metadata["vocab_size"],
        d_model=metadata["d_model"],
        num_layers=metadata["num_layers"],
        num_heads=metadata["num_heads"]
    )

    # -------------------------------------------------------
    # Create full model and load weights
    # -------------------------------------------------------
    model = HMERModel(encoder, decoder, attention).to(device)
    model.load_state_dict(state_dict)

    # -------------------------------------------------------
    # Collect sample images (.png, .jpg, .jpeg)
    # -------------------------------------------------------
    img_dir = os.path.dirname(__file__)
    patterns = ["sample*.png", "sample*.jpg", "sample*.jpeg"]

    image_paths = []
    for p in patterns:
        image_paths.extend(glob.glob(os.path.join(img_dir, p)))

    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise FileNotFoundError(
            "No sample images found. Add sample.jpg or sample.png near inference.py."
        )

    print(f"\nFound {len(image_paths)} images to process:")

    for p in image_paths:
        print(" -", os.path.basename(p))

    # -------------------------------------------------------
    # Run inference on each image
    # -------------------------------------------------------
    for path in image_paths:
        print(f"\nProcessing {os.path.basename(path)}...")

        img_tensor, pil_img = preprocess_image(path)
        decoded, attn = generate_sequence(model, tokenizer, img_tensor, device)

        print("Predicted LaTeX:", " ".join(decoded))

        # visualize attention of last token
        visualize_attention(pil_img, attn[0].cpu().numpy(), decoded[-1])


if __name__ == "__main__":
    main()