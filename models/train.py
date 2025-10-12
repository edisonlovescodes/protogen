"""
Training script for Protein Transformer model.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from transformer import ProteinTransformer
import sys


class ProteinSequenceDataset(Dataset):
    """Dataset for protein sequences."""
    
    def __init__(self, data_path: str, max_length: int = 256):
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.sequences = data['sequences']
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_data = self.sequences[idx]
        tokens = seq_data['tokens'][:self.max_length]
        
        # Pad if needed
        if len(tokens) < self.max_length:
            tokens = tokens + [20] * (self.max_length - len(tokens))  # 20 is PAD token
        
        return torch.tensor(tokens, dtype=torch.long)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        
        # Shift input and target
        input_ids = batch[:, :-1]
        target_ids = batch[:, 1:]
        
        # Forward pass
        logits = model(input_ids)
        loss = criterion(logits.reshape(-1, model.vocab_size), target_ids.reshape(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]
            
            logits = model(input_ids)
            loss = criterion(logits.reshape(-1, model.vocab_size), target_ids.reshape(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    print("Training Protein Transformer")
    print("=" * 50)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    vocab_size = 24
    d_model = 128
    num_layers = 2
    batch_size = 8
    learning_rate = 1e-3
    num_epochs = 3
    max_seq_length = 256
    
    # Create datasets
    print("Loading data...")
    train_dataset = ProteinSequenceDataset("data/processed/tokenized_sequences.json", max_seq_length)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")
    
    # Create model
    print("Creating model...")
    model = ProteinTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        max_seq_length=max_seq_length
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=20)  # Ignore PAD token
    
    # Training loop
    print("Starting training...\n")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")
    
    # Save model
    save_path = "models/checkpoint.pt"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    # Generate sample
    print("\nGenerating sample sequence...")
    model.eval()
    generated = model.generate(start_token=22, max_length=50, device=device)
    print(f"Generated tokens: {generated}")


if __name__ == "__main__":
    main()
