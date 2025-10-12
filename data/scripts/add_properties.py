"""
Add property labels to protein sequences.
We'll use simple heuristics to assign properties.
"""

import json
from pathlib import Path


def assign_properties(sequence: str) -> dict:
    """
    Assign properties to a protein sequence based on composition.
    
    Args:
        sequence: Protein sequence
        
    Returns:
        Dictionary of properties with scores 0-1
    """
    sequence = sequence.upper()
    length = len(sequence)
    
    # Count amino acid types
    hydrophobic = sum(1 for aa in sequence if aa in 'AILMFVP')
    charged = sum(1 for aa in sequence if aa in 'DEKR')
    polar = sum(1 for aa in sequence if aa in 'STCN')
    aromatic = sum(1 for aa in sequence if aa in 'FWY')
    
    # Calculate ratios
    hydrophobic_ratio = hydrophobic / length
    charged_ratio = charged / length
    polar_ratio = polar / length
    aromatic_ratio = aromatic / length
    
    # Assign properties based on composition
    properties = {
        'thermostable': min(1.0, (hydrophobic_ratio * 1.5 + aromatic_ratio) / 2),
        'enzyme': min(1.0, (charged_ratio * 1.2 + polar_ratio) / 2),
        'membrane': min(1.0, hydrophobic_ratio * 2),
        'structural': min(1.0, (aromatic_ratio + polar_ratio) / 2),
        'flexible': min(1.0, polar_ratio * 2)
    }
    
    return properties


def main():
    print("Adding property labels to sequences...\n")
    
    # Load processed sequences
    with open("data/processed/tokenized_sequences.json", 'r') as f:
        data = json.load(f)
    
    sequences = data['sequences']
    
    # Add properties to each sequence
    for seq_data in sequences:
        original_seq = seq_data['original_sequence']
        properties = assign_properties(original_seq)
        seq_data['properties'] = properties
    
    # Save with properties
    with open("data/processed/sequences_with_properties.json", 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Added properties to {len(sequences)} sequences")
    print("\nProperty Examples:")
    
    for i in range(min(3, len(sequences))):
        seq = sequences[i]
        print(f"\nSequence {i+1}: {seq['original_sequence'][:50]}...")
        print(f"  Properties: {seq['properties']}")


if __name__ == "__main__":
    main()
