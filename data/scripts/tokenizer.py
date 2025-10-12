"""
Protogen Phase 1: Amino Acid Tokenizer
A simple tokenizer to convert protein sequences into numerical vectors.
"""

import json
from pathlib import Path
from typing import Dict, List


class AminoAcidTokenizer:
    """Convert amino acid sequences to token indices and back."""
    
    # Standard 20 amino acids
    AMINO_ACIDS = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
        'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'
    ]
    
    def __init__(self):
        """Initialize the tokenizer."""
        self.aa_to_token = {aa: i for i, aa in enumerate(self.AMINO_ACIDS)}
        
        # Special tokens
        self.aa_to_token['<PAD>'] = len(self.AMINO_ACIDS)
        self.aa_to_token['<UNK>'] = len(self.AMINO_ACIDS) + 1
        self.aa_to_token['<START>'] = len(self.AMINO_ACIDS) + 2
        self.aa_to_token['<END>'] = len(self.AMINO_ACIDS) + 3
        
        self.token_to_aa = {v: k for k, v in self.aa_to_token.items()}
        self.vocab_size = len(self.aa_to_token)
    
    def encode(self, sequence: str) -> List[int]:
        """Convert amino acid sequence to token indices."""
        tokens = [self.aa_to_token['<START>']]
        
        for aa in sequence.upper():
            if aa in self.aa_to_token:
                tokens.append(self.aa_to_token[aa])
            else:
                tokens.append(self.aa_to_token['<UNK>'])
        
        tokens.append(self.aa_to_token['<END>'])
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Convert token indices back to amino acid sequence."""
        sequence = []
        
        for token in tokens:
            if token in self.token_to_aa:
                aa = self.token_to_aa[token]
                if aa not in ['<PAD>', '<START>', '<END>', '<UNK>']:
                    sequence.append(aa)
        
        return ''.join(sequence)
    
    def save(self, path: str):
        """Save tokenizer configuration to JSON."""
        config = {
            'aa_to_token': self.aa_to_token,
            'vocab_size': self.vocab_size
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load tokenizer from JSON."""
        tokenizer = cls()
        with open(path, 'r') as f:
            config = json.load(f)
        tokenizer.aa_to_token = config['aa_to_token']
        tokenizer.token_to_aa = {int(v): k for k, v in tokenizer.aa_to_token.items()}
        tokenizer.vocab_size = config['vocab_size']
        return tokenizer
    
    def pad_sequence(self, tokens: List[int], max_length: int) -> List[int]:
        """Pad sequence to fixed length."""
        pad_token = self.aa_to_token['<PAD>']
        
        if len(tokens) > max_length:
            return tokens[:max_length]
        else:
            return tokens + [pad_token] * (max_length - len(tokens))


if __name__ == "__main__":
    tokenizer = AminoAcidTokenizer()
    
    print("Protogen: Amino Acid Tokenizer")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Amino acids: {', '.join(tokenizer.AMINO_ACIDS)}\n")
    
    # Test
    test_seq = "MVHLTPEEKS"
    tokens = tokenizer.encode(test_seq)
    decoded = tokenizer.decode(tokens)
    
    print(f"Original: {test_seq}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {decoded}")