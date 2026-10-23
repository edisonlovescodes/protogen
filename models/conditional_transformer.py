"""
Conditional Transformer - generates proteins with specific properties.
"""

import torch
import torch.nn as nn

from .transformer import ProteinTransformer


class ConditionalProteinTransformer(ProteinTransformer):
    """Causal Transformer conditioned on five composition proxies."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int = 1024,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        num_properties: int = 5,
    ):
        super().__init__(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_length, dropout)

        self.num_properties = num_properties

        self.property_embedding = nn.Linear(num_properties, d_model)

    def forward(
        self,
        x: torch.Tensor,
        properties: torch.Tensor,
    ) -> torch.Tensor:
        mask, padding_mask = self.attention_masks(x)
        x = self.embedding(x) * (self.d_model**0.5)
        x = self.pos_encoding(x)

        x[:, 0, :] += self.property_embedding(properties)

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask, padding_mask)

        return self.output(x)

    def generate_with_properties(
        self, properties: list, max_length: int, temperature: float
    ) -> list:
        """Sample residues from a causal prefix until END or the token limit."""
        if not 0 < temperature <= 2 or not 2 <= max_length <= self.pos_encoding.pe.shape[1]:
            raise ValueError("Invalid temperature or maximum length")
        if len(properties) != self.num_properties or not all(0 <= p <= 1 for p in properties):
            raise ValueError("Properties must be five proxy scores in [0, 1]")
        self.eval()

        device = self.embedding.weight.device
        tokens = [22]  # START token

        prop_tensor = torch.tensor(properties, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            for _ in range(max_length - 1):
                input_ids = torch.tensor([tokens]).to(device)

                logits = self.forward(input_ids, properties=prop_tensor)

                next_logits = logits[0, -1, :] / temperature
                next_logits[20:23] = -torch.inf
                if len(tokens) == 1:
                    next_logits[23] = -torch.inf  # require at least one residue
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                tokens.append(next_token)

                if next_token == 23:  # END token
                    break

        return tokens
