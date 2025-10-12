"""
Train improved model with more data and longer training.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from conditional_transformer import ConditionalProteinTransformer


class ConditionalProteinDataset(Dataset):
    """Dataset with property conditioning."""
    
    def __init__(self, data_path: str, max_length: int = 512):
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        self.sequences = data['sequences']
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_data = self.sequences[idx]
        tokens = seq_data['tokens'][:self.max_length]
        
        if len(tokens) < self.max_length:
            tokens = tokens + [20] * (self.max_length - len(tokens))
        
        props = seq_data.get('properties', {
            'thermostable': 0.5, 'enzyme': 0.5, 'membrane': 0.5,
            'structural': 0.5, 'flexible': 0.5
        })
        
        prop_values = [
            props['thermostable'], props['enzyme'], props['membrane'],
            props['structural'], props['flexible']
        ]
        
        return {
            'tokens': torch.tensor(tokens, dtype=torch.long),
            'properties': torch.tensor(prop_values, dtype=torch.float32)
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        tokens = batch['tokens'].to(device)
        properties = batch['properties'].to(device)
        
        input_ids = tokens[:, :-1]
        target_ids = tokens[:, 1:]
        
        logits = model(input_ids, properties=properties)
        loss = criterion(logits.reshape(-1, model.vocab_size), target_ids.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: Loss = {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            tokens = batch['tokens'].to(device)
            properties = batch['properties'].to(device)
            
            input_ids = tokens[:, :-1]
            target_ids = tokens[:, 1:]
            
            logits = model(input_ids, properties=properties)
            loss = criterion(logits.reshape(-1, model.vocab_size), target_ids.reshape(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    print("=" * 60)
    print("Training IMPROVED Conditional Protein Transformer")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Larger model configuration
    vocab_size = 24
    d_model = 256  # Increased from 128
    num_layers = 4  # Increased from 2
    batch_size = 16  # Increased from 8
    learning_rate = 5e-4  # Slightly lower
    num_epochs = 10  # Increased from 3
    max_seq_length = 512
    
    print("Loading data...")
    dataset = ConditionalProteinDataset("data/processed/large_dataset_processed.json", max_seq_length)
    
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")
    
    print("Creating LARGER model...")
    model = ConditionalProteinTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        max_seq_length=max_seq_length,
        num_properties=5
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=20)
    
    best_val_loss = float('inf')
    
    print("Starting training for 10 epochs...\n")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 60)
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/best_model.pt")
            print(f"✓ Best model saved! (Val Loss: {val_loss:.4f})")
        
        print()
    
    # Save final model
    torch.save(model.state_dict(), "models/final_model.pt")
    print(f"\n✅ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved: models/best_model.pt, models/final_model.pt")
    
    # Test generation
    print("\n" + "=" * 60)
    print("Testing improved model generation...")
    print("=" * 60)
    
    model.eval()
    
    test_cases = [
        ([0.95, 0.1, 0.1, 0.5, 0.2], "Highly Thermostable"),
        ([0.1, 0.95, 0.1, 0.5, 0.8], "Highly Enzyme-like"),
        ([0.2, 0.2, 0.95, 0.3, 0.2], "Membrane Protein"),
    ]
    
    for props, desc in test_cases:
        print(f"\n{desc} (props: {props}):")
        generated = model.generate_with_properties(props, max_length=80, device=device)
        print(f"  Length: {len(generated)} tokens")
        print(f"  Tokens: {generated[:20]}...")


if __name__ == "__main__":
    main()
