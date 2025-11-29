from torchvision import transforms
from models.math_writing_dataset_loader import MathWritingDataLoader
from models.latex_tokenizer import LaTeXCharTokenizer


def train():
    transform = transforms.Compose([
        transforms.Resize((128, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.Grayscale(num_output_channels=1),
    ])

    data_loader = MathWritingDataLoader(split="train", transform=transform, tokenizer=LaTeXCharTokenizer())

    encoder = DenseNetEncoder()
    

if __name__ == "__main__":
    train()