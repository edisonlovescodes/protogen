"""
Simple Transformer model for protein sequence generation.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Add positional encoding to embeddings."""

    def __init__(self, d_model: int, max_seq_length: int = 512):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, : x.size(1), :]


class TransformerBlock(nn.Module):
    """Single Transformer block with attention and feed-forward."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Self-attention
        attn_out, _ = self.attention(
            x, x, x, attn_mask=mask, key_padding_mask=padding_mask, need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


class ProteinTransformer(nn.Module):
    """Transformer model for protein sequence generation."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        max_seq_length: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Output layer
        self.output = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention_masks(tokens, mask=None):
        # True means blocked for MultiheadAttention. Future tokens are always blocked.
        length = tokens.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=tokens.device).triu(1)
        if mask is not None:
            if mask.dtype != torch.bool:
                raise ValueError("Use a boolean attention mask")
            causal = causal | mask.to(tokens.device)
        return causal, tokens.eq(20)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input token indices (batch_size, seq_length)
            mask: Attention mask

        Returns:
            Logits (batch_size, seq_length, vocab_size)
        """
        mask, padding_mask = self.attention_masks(x, mask)
        # Embed tokens
        x = self.embedding(x) * (self.d_model**0.5)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask, padding_mask)

        # Output layer
        logits = self.output(x)

        return logits

    def generate(
        self, start_token: int, max_length: int = 100, temperature: float = 1.0, device: str = "cpu"
    ) -> list:
        """
        Generate protein sequence greedily.

        Args:
            start_token: Starting token index
            max_length: Maximum sequence length
            temperature: Sampling temperature
            device: Device to use

        Returns:
            List of generated token indices
        """
        if not 0 < temperature <= 2 or not 2 <= max_length <= self.pos_encoding.pe.shape[1]:
            raise ValueError("Invalid temperature or maximum length")
        self.eval()

        tokens = [start_token]

        with torch.no_grad():
            for _ in range(max_length - 1):
                # Convert to tensor
                input_ids = torch.tensor([tokens]).to(device)

                # Forward pass
                logits = self.forward(input_ids)

                # Get last token logits
                next_logits = logits[0, -1, :] / temperature

                # Sample from distribution
                next_logits[20:23] = -torch.inf  # do not generate PAD/UNK/START
                if len(tokens) == 1:
                    next_logits[23] = -torch.inf  # require at least one residue
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                tokens.append(next_token)

                # Stop if end token
                if next_token == 23:  # End token
                    break

        return tokens


if __name__ == "__main__":
    print("Protein Transformer Model")
    print("-" * 40)

    # Initialize model
    vocab_size = 24  # 20 amino acids + 4 special tokens
    model = ProteinTransformer(vocab_size=vocab_size, d_model=128, num_layers=2)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size, seq_length = 4, 50
    x = torch.randint(0, vocab_size, (batch_size, seq_length))

    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Test generation
    print("\nGenerating sequence...")
    generated = model.generate(start_token=22, max_length=30)
    print(f"Generated tokens: {generated}")
