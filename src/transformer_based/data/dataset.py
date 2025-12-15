import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as T

class MathWritingDataset(Dataset):
    def __init__(self, split, tokenizer, img_size=320):
        self.ds = load_dataset("deepcopy/MathWriting-human")[split]
        limit = int(0.1 * len(self.ds))  # Use 10% of data for faster experiments
        self.ds = self.ds.select(range(limit))
        self.tokenizer = tokenizer

        self.transform = T.Compose([
            T.Grayscale(),
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        
        img = item["image"]
        if not isinstance(img, Image.Image):
            # fallback if dataset ever changes
            img = Image.open(img)

        img = img.convert("L")
        img = self.transform(img)

        tokens = self.tokenizer.encode(item["latex"])

        return {
            "image": img,
            "tokens": torch.tensor(tokens, dtype=torch.long),
        }