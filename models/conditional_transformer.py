"""
Conditional Transformer - generates proteins with specific properties.
"""

import torch
import torch.nn as nn
from typing import Optional

try:
    from .transformer import ProteinTransformer
except ImportError:
    from transformer import ProteinTransformer


class ConditionalProteinTransformer(ProteinTransformer):
    """Transformer conditioned on protein properties."""

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

        # Property embedding layer
        self.property_embedding = nn.Linear(num_properties, d_model)

    def forward(
        self,
        x: torch.Tensor,
        properties: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional property conditioning.

        Args:
            x: Input token indices (batch_size, seq_length)
            properties: Property values (batch_size, num_properties)
            mask: Attention mask

        Returns:
            Logits (batch_size, seq_length, vocab_size)
        """
        mask, padding_mask = self.attention_masks(x, mask)
        # Embed tokens
        x = self.embedding(x) * (self.d_model**0.5)
        x = self.pos_encoding(x)

        # Add property conditioning if provided
        if properties is not None:
            # Embed properties
            prop_embed = self.property_embedding(properties)  # (batch_size, d_model)

            # Add to first position of each sequence
            x[:, 0, :] = x[:, 0, :] + prop_embed

        x = self.dropout(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask, padding_mask)

        # Output layer
        logits = self.output(x)

        return logits

    def generate_with_properties(
        self, properties: list, max_length: int = 100, temperature: float = 1.0, device: str = "cpu"
    ) -> list:
        """
        Generate protein with specific properties.

        Args:
            properties: List of property values [thermostable, enzyme, membrane, structural, flexible]
            max_length: Maximum sequence length
            temperature: Sampling temperature
            device: Device to use

        Returns:
            List of generated token indices
        """
        if not 0 < temperature <= 2 or not 2 <= max_length <= self.pos_encoding.pe.shape[1]:
            raise ValueError("Invalid temperature or maximum length")
        if len(properties) != self.num_properties or not all(0 <= p <= 1 for p in properties):
            raise ValueError("Properties must be five proxy scores in [0, 1]")
        self.eval()

        tokens = [22]  # START token

        # Convert properties to tensor
        prop_tensor = torch.tensor(properties, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            for _ in range(max_length - 1):
                input_ids = torch.tensor([tokens]).to(device)

                # Forward pass with properties
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


if __name__ == "__main__":
    print("Conditional Protein Transformer")
    print("-" * 40)

    vocab_size = 24
    model = ConditionalProteinTransformer(vocab_size=vocab_size, d_model=128, num_layers=2)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with properties
    batch_size, seq_length = 4, 50
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    props = torch.randn(batch_size, 5)  # 5 properties

    output = model(x, properties=props)
    print(f"Input shape: {x.shape}")
    print(f"Properties shape: {props.shape}")
    print(f"Output shape: {output.shape}")

    # Test generation
    print("\nGenerating with properties...")
    test_props = [0.8, 0.2, 0.1, 0.5, 0.3]  # Thermostable, low enzyme, etc.
    generated = model.generate_with_properties(test_props, max_length=50)
    print(f"Generated: {generated}")
