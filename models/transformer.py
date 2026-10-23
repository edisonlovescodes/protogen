"""
Simple Transformer model for protein sequence generation.
"""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Add positional encoding to embeddings."""

    def __init__(self, d_model: int, max_seq_length: int = 512):
        super().__init__()

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
        mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with residual connections."""
        attn_out, _ = self.attention(
            x, x, x, attn_mask=mask, key_padding_mask=padding_mask, need_weights=False
        )
        x = self.norm1(x + self.dropout(attn_out))

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

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.output = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention_masks(tokens):
        length = tokens.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=tokens.device).triu(1)
        return causal, tokens.eq(20)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask, padding_mask = self.attention_masks(x)
        x = self.embedding(x) * (self.d_model**0.5)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask, padding_mask)

        return self.output(x)

    def generate(self, max_length: int, temperature: float) -> list:
        """Sample residues from a causal prefix until END or the token limit."""
        if not 0 < temperature <= 2 or not 2 <= max_length <= self.pos_encoding.pe.shape[1]:
            raise ValueError("Invalid temperature or maximum length")
        self.eval()

        tokens = [22]

        with torch.no_grad():
            for _ in range(max_length - 1):
                input_ids = torch.tensor([tokens], device=self.embedding.weight.device)

                logits = self.forward(input_ids)

                next_logits = logits[0, -1, :] / temperature

                next_logits[20:23] = -torch.inf  # do not generate PAD/UNK/START
                if len(tokens) == 1:
                    next_logits[23] = -torch.inf  # require at least one residue
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                tokens.append(next_token)

                if next_token == 23:  # End token
                    break

        return tokens
