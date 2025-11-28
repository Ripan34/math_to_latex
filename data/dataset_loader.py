from datasets import load_dataset

def load_mathwriting(split="train"):
    ds = load_dataset("deepcopy/MathWriting-Human")
    return ds[split]
