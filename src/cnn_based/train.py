from torchvision import transforms
from models.math_writing_dataset_loader import MathWritingDataLoader
from models.latex_tokenizer import LaTeXCharTokenizer
from models.encoder_densenet import DenseNetEncoder
from models.decoder import LSTMDecoder
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


def greedy_decode(encoder, decoder, image, tokenizer, max_length=150, device='cpu'):

    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Extract features
        features = encoder(image.unsqueeze(0).to(device))
        
        current_token = torch.tensor([[tokenizer.char_to_idx[tokenizer.START_TOKEN]]], 
                                     dtype=torch.long, device=device)
        generated_tokens = [current_token.item()]
        
        hidden = None
        
        for _ in range(max_length):
            embeddings = decoder.embedding(current_token)
            
            if hidden is None:
                h = torch.zeros(1, decoder.lstm.hidden_size).to(device)
            else:
                h = hidden[0].squeeze(0)
            
            hidden_proj = decoder.attn_hidden(h).unsqueeze(1)
            enc_proj = decoder.attn_enc(features)
            scores = decoder.attn_score(torch.tanh(enc_proj + hidden_proj))
            scores = scores.squeeze(-1)
            attn_weights = F.softmax(scores, dim=1)
            context = torch.bmm(attn_weights.unsqueeze(1), features)
            
            # LSTM forward
            lstm_input = torch.cat([embeddings, context], dim=-1)
            lstm_out, hidden = decoder.lstm(lstm_input, hidden)
            
            output = decoder.fc(lstm_out)
            next_token = output.argmax(dim=-1)
            
            next_token_id = next_token.item()
            generated_tokens.append(next_token_id)
            
            if next_token_id == tokenizer.char_to_idx[tokenizer.END_TOKEN]:
                break
            
            current_token = next_token
    
    return generated_tokens


def calculate_cer(predicted, target):

    if len(predicted) == 0:
        return len(target)
    if len(target) == 0:
        return len(predicted)
    
    # Create distance matrix
    d = [[0] * (len(target) + 1) for _ in range(len(predicted) + 1)]
    
    for i in range(len(predicted) + 1):
        d[i][0] = i
    for j in range(len(target) + 1):
        d[0][j] = j
    
    for i in range(1, len(predicted) + 1):
        for j in range(1, len(target) + 1):
            if predicted[i-1] == target[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    
    return d[len(predicted)][len(target)] / max(len(target), 1)


def evaluate(encoder, decoder, val_loader, tokenizer, device='cpu', num_samples=5):
    """
    Evaluate the model on validation set.
    Returns: average CER, exact match accuracy, and sample predictions.
    """
    encoder.eval()
    decoder.eval()
    
    total_cer = 0.0
    exact_matches = 0
    total_samples = 0
    samples_to_show = []
    
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images = batch["images"].to(device)
            target_texts = batch["latex"]
            
            for i in range(images.size(0)):
                # Generate prediction
                predicted_tokens = greedy_decode(
                    encoder, decoder, images[i], tokenizer, device=device
                )
                predicted_text = tokenizer.decode(predicted_tokens)
                target_text = target_texts[i]
                
                # Calculate metrics
                cer = calculate_cer(predicted_text, target_text)
                total_cer += cer
                
                if predicted_text == target_text:
                    exact_matches += 1
                
                total_samples += 1
                
                if len(samples_to_show) < num_samples:
                    samples_to_show.append({
                        'predicted': predicted_text,
                        'target': target_text,
                        'cer': cer
                    })
    
    avg_cer = total_cer / max(total_samples, 1)
    accuracy = exact_matches / max(total_samples, 1)
    
    print(f"\nOverall Metrics:")
    print(f"  Average CER: {avg_cer:.4f}")
    print(f"  Exact Match Accuracy: {accuracy:.4f} ({exact_matches}/{total_samples})")
    
    print(f"\nSample Predictions:")
    print("-" * 80)
    for idx, sample in enumerate(samples_to_show, 1):
        print(f"\nSample {idx}:")
        print(f"  Target:    {sample['target']}")
        print(f"  Predicted: {sample['predicted']}")
        print(f"  CER:       {sample['cer']:.4f}")
    
    print("="*80 + "\n")
    
    return avg_cer, accuracy

def train():
    device = 'cpu'
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((128, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tokenizer = LaTeXCharTokenizer()
    
    train_ds = MathWritingDataLoader(
        split="train", 
        transform=transform, 
        tokenizer=tokenizer,
        max_samples=2000
    )
    
    # Add validation dataset
    val_ds = MathWritingDataLoader(
        split="val",
        transform=transform,
        tokenizer=tokenizer,
        max_samples=50
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=mathwriting_collate_fn
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=mathwriting_collate_fn
    )
    
    encoder = DenseNetEncoder().to(device)
    
    decoder = LSTMDecoder(
        vocab_size=tokenizer.vocab_size(),
        embed_dim=256,
        hidden_dim=512,
        encoder_dim=encoder.output_dim
    ).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-4
    )

    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(10):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/10")
        print(f"{'='*80}")
        
        encoder.train()
        decoder.train()
        
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch["images"].to(device)
            captions = batch["token_ids"].to(device)

            # 1. Extract features from encoder
            features = encoder(images)

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            # 3. Decoder forward pass
            outputs = decoder(features, inputs)   # (B, seq_len-1, vocab)

            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"Average Training Loss: {avg_loss:.4f}")
        
    # Evaluate
    evaluate(encoder, decoder, val_loader, tokenizer, device=device, num_samples=3)
    print("\nTraining completed")
    
    return encoder, decoder, tokenizer



def mathwriting_collate_fn(batch):
    images = []
    token_seqs = []
    latex_texts = []

    for item in batch:
        images.append(item["image"])
        token_seqs.append(torch.tensor(item["tokenized_input"], dtype=torch.long))
        latex_texts.append(item["latex"])

    images = torch.stack(images)

    padded_tokens = pad_sequence(token_seqs, batch_first=True, padding_value=0)

    return {
        "images": images,
        "token_ids": padded_tokens,
        "latex": latex_texts
    }

if __name__ == "__main__":
    encoder, decoder, tokenizer = train()
    
    # Save the trained models
    print("\nSaving models...")
    torch.save(encoder.state_dict(), "encoder_checkpoint.pth")
    torch.save(decoder.state_dict(), "decoder_checkpoint.pth")
    print("Models saved as encoder_checkpoint.pth and decoder_checkpoint.pth")