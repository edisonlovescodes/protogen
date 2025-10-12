"""
Process the larger dataset with properties.
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tokenizer import AminoAcidTokenizer


def assign_properties(sequence: str) -> dict:
    """Assign properties based on composition."""
    sequence = sequence.upper()
    length = len(sequence)
    
    hydrophobic = sum(1 for aa in sequence if aa in 'AILMFVP')
    charged = sum(1 for aa in sequence if aa in 'DEKR')
    polar = sum(1 for aa in sequence if aa in 'STCN')
    aromatic = sum(1 for aa in sequence if aa in 'FWY')
    
    hydrophobic_ratio = hydrophobic / length
    charged_ratio = charged / length
    polar_ratio = polar / length
    aromatic_ratio = aromatic / length
    
    return {
        'thermostable': min(1.0, (hydrophobic_ratio * 1.5 + aromatic_ratio) / 2),
        'enzyme': min(1.0, (charged_ratio * 1.2 + polar_ratio) / 2),
        'membrane': min(1.0, hydrophobic_ratio * 2),
        'structural': min(1.0, (aromatic_ratio + polar_ratio) / 2),
        'flexible': min(1.0, polar_ratio * 2)
    }


def main():
    print("Processing large dataset...\n")
    
    # Load raw sequences
    with open("data/raw/large_dataset.json", 'r') as f:
        data = json.load(f)
    
    sequences = data['sequences']
    print(f"Loaded {len(sequences)} sequences")
    
    # Initialize tokenizer
    tokenizer = AminoAcidTokenizer()
    
    # Process each sequence
    processed = []
    max_length = 512
    
    for i, seq in enumerate(sequences):
        tokens = tokenizer.encode(seq)
        padded = tokenizer.pad_sequence(tokens, max_length)
        properties = assign_properties(seq)
        
        processed.append({
            'original_sequence': seq,
            'tokens': padded,
            'original_length': len(seq),
            'token_length': len(tokens),
            'properties': properties
        })
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(sequences)}...")
    
    # Save processed data
    output_data = {
        'sequences': processed,
        'count': len(processed)
    }
    
    with open("data/processed/large_dataset_processed.json", 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved {len(processed)} processed sequences")
    print("Ready for training!")


if __name__ == "__main__":
    main()
