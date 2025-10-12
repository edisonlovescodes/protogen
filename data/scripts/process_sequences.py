"""
Process protein sequences for model training.
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tokenizer import AminoAcidTokenizer


def load_sequences(input_path: str) -> list:
    """Load sequences from JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    return data['sequences']


def process_sequences(sequences: list, tokenizer: AminoAcidTokenizer, max_length: int = 512) -> list:
    """
    Process sequences into tokens.
    
    Args:
        sequences: List of protein sequences
        tokenizer: AminoAcidTokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        List of tokenized sequences
    """
    processed = []
    
    for i, seq in enumerate(sequences):
        # Encode sequence
        tokens = tokenizer.encode(seq)
        
        # Pad or truncate
        padded = tokenizer.pad_sequence(tokens, max_length)
        
        processed.append({
            'original_sequence': seq,
            'tokens': padded,
            'original_length': len(seq),
            'token_length': len(tokens)
        })
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(sequences)} sequences...")
    
    return processed


def save_processed_data(processed_sequences: list, output_path: str):
    """Save processed sequences to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'sequences': processed_sequences,
        'count': len(processed_sequences)
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(processed_sequences)} processed sequences to {output_path}")


def main():
    print("Processing protein sequences for training...\n")
    
    # Initialize tokenizer
    tokenizer = AminoAcidTokenizer()
    
    # Load raw sequences
    print("Loading sequences from data/raw/sample_proteins.json...")
    sequences = load_sequences("data/raw/sample_proteins.json")
    print(f"Loaded {len(sequences)} sequences\n")
    
    # Process sequences
    print("Tokenizing sequences...")
    processed = process_sequences(sequences, tokenizer, max_length=512)
    
    # Save processed data
    print()
    save_processed_data(processed, "data/processed/tokenized_sequences.json")
    
    # Print statistics
    print("\nProcessed Data Statistics:")
    original_lengths = [s['original_length'] for s in processed]
    print(f"  Total sequences: {len(processed)}")
    print(f"  Average original length: {sum(original_lengths) / len(original_lengths):.0f} AA")
    print(f"  Min length: {min(original_lengths)} AA")
    print(f"  Max length: {max(original_lengths)} AA")
    
    # Save tokenizer
    tokenizer.save("data/tokenizer.json")
    print("\nTokenizer saved to data/tokenizer.json")


if __name__ == "__main__":
    main()
