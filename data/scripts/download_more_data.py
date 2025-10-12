"""
Download more protein sequences for better training.
"""

import requests
import json
from pathlib import Path
from tqdm import tqdm


def download_proteins(num_sequences: int = 500):
    """Download proteins from UniProt."""
    print(f"Downloading {num_sequences} protein sequences from UniProt...")
    
    sequences = []
    batch_size = 100
    
    for offset in tqdm(range(0, num_sequences, batch_size)):
        url = "https://rest.uniprot.org/uniprotkb/search"
        
        params = {
            "query": "reviewed:true AND organism_id:9606",
            "format": "json",
            "size": min(batch_size, num_sequences - offset),
            "fields": "sequence",
            "offset": offset
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'results' in data:
                for result in data['results']:
                    if 'sequence' in result:
                        seq = result['sequence'].get('value', '')
                        if 50 < len(seq) < 1000:  # Reasonable length
                            sequences.append(seq)
        
        except Exception as e:
            print(f"Error at offset {offset}: {e}")
            continue
    
    print(f"Downloaded {len(sequences)} sequences")
    return sequences


def save_sequences(sequences, output_path):
    """Save sequences to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'sequences': sequences,
        'count': len(sequences)
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    sequences = download_proteins(num_sequences=500)
    save_sequences(sequences, "data/raw/large_dataset.json")
    
    lengths = [len(seq) for seq in sequences]
    print(f"\nDataset Statistics:")
    print(f"  Total: {len(sequences)}")
    print(f"  Avg length: {sum(lengths) / len(lengths):.0f} AA")
    print(f"  Min: {min(lengths)} AA")
    print(f"  Max: {max(lengths)} AA")
