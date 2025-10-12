"""
Download sample protein sequences from UniProt.
We'll fetch a small subset for testing.
"""

import requests
import json
from pathlib import Path


def download_sample_proteins(num_sequences: int = 100) -> list:
    """
    Download sample protein sequences from UniProt API.
    
    Args:
        num_sequences: Number of sequences to download
        
    Returns:
        List of protein sequences
    """
    print(f"Downloading {num_sequences} sample protein sequences from UniProt...")
    
    sequences = []
    
    # UniProt REST API query
    # Getting reviewed human proteins, sorting by length
    url = "https://rest.uniprot.org/uniprotkb/search"
    
    params = {
        "query": "reviewed:true AND organism_id:9606",  # Human proteins only
        "format": "json",
        "size": num_sequences,
        "fields": "sequence"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'results' in data:
            for result in data['results']:
                if 'sequence' in result:
                    seq = result['sequence'].get('value', '')
                    if len(seq) > 50:  # Only keep sequences longer than 50 AA
                        sequences.append(seq)
        
        print(f"Downloaded {len(sequences)} sequences")
        return sequences
    
    except Exception as e:
        print(f"Error downloading from UniProt: {e}")
        print("Using example sequences instead...")
        return get_example_sequences()


def get_example_sequences() -> list:
    """Return example protein sequences if download fails."""
    return [
        "MVHLTPEEKS",
        "MSPILGYWKIKGLVQPTRLLLEYLLEEKGDLLEVKKFVAKAKGSFTVFKAKLTVAEALHHPVLEAFDGDKDQILKK",
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV",
        "MGLSDGEWQQVLNVWGKVEADIPGHGQEVLIRPFLKTPQAKTFSFPDLSPPDAFTPVNSAAYAQDFFEAKGQKLVADIG",
        "MHIQAPILGYYSGDDQKDTVIVAPNKSGKIAKYSNSIKNIPMPPVVPPGAKD",
    ]


def save_sequences(sequences: list, output_path: str):
    """Save sequences to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'sequences': sequences,
        'count': len(sequences)
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(sequences)} sequences to {output_path}")


if __name__ == "__main__":
    # Download sequences
    sequences = download_sample_proteins(num_sequences=50)
    
    # Save to file
    save_sequences(sequences, "data/raw/sample_proteins.json")
    
    # Print stats
    print("\nSequence Statistics:")
    lengths = [len(seq) for seq in sequences]
    print(f"  Total sequences: {len(sequences)}")
    print(f"  Average length: {sum(lengths) / len(lengths):.0f} AA")
    print(f"  Min length: {min(lengths)} AA")
    print(f"  Max length: {max(lengths)} AA")
