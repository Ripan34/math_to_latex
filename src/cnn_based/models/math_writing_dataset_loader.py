from torch.utils.data import Dataset
from datasets import load_dataset

class MathWritingDataLoader(Dataset):
    def __init__(self, split="train", tokenizer=None, transform=None):
        self.data = load_dataset("deepcopy/MathWriting-Human")[split]
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = item["image"]
        latex = item["latex"]

        if self.transform:
            image = self.transform(image)

        if self.tokenizer:
            tokenized_input = self.tokenizer.encode(latex)
        else:
            tokenized_input = latex

        return {
            "image": image,
            "tokenized_input": tokenized_input,
            "latex": latex,
        }
