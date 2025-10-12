"""
FastAPI backend for Protogen with property conditioning.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "scripts"))

from conditional_transformer import ConditionalProteinTransformer
from tokenizer import AminoAcidTokenizer

app = FastAPI(title="Protogen API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading model...")
tokenizer = AminoAcidTokenizer.load("data/tokenizer.json")
model = ConditionalProteinTransformer(vocab_size=24, d_model=256, num_layers=4, max_seq_length=512, num_properties=5)
model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
model = model.to(device)
model.eval()

print("Conditional model loaded!")


class GenerateRequest(BaseModel):
    max_length: int = 100
    temperature: float = 1.0
    num_sequences: int = 1  # Change this line
    properties: dict = {
        'thermostable': 0.5,
        'enzyme': 0.5,
        'membrane': 0.5,
        'structural': 0.5,
        'flexible': 0.5
    }


class GenerateResponse(BaseModel):
    sequences: list
    tokens: list
    properties: dict


@app.get("/")
def root():
    return {
        "name": "Protogen API",
        "version": "2.0.0",
        "description": "Generate proteins with specific properties using AI"
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model": "conditional"}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    try:
        sequences = []
        tokens_list = []
        
        # Convert properties dict to list
        prop_values = [
            request.properties.get('thermostable', 0.5),
            request.properties.get('enzyme', 0.5),
            request.properties.get('membrane', 0.5),
            request.properties.get('structural', 0.5),
            request.properties.get('flexible', 0.5)
        ]
        
        for _ in range(request.num_sequences):
            generated_tokens = model.generate_with_properties(
                properties=prop_values,
                max_length=request.max_length,
                temperature=request.temperature,
                device=device
            )
            
            sequence = tokenizer.decode(generated_tokens)
            sequences.append(sequence)
            tokens_list.append(generated_tokens)
        
        return GenerateResponse(
            sequences=sequences,
            tokens=tokens_list,
            properties=request.properties
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/properties")
def get_properties():
    """Get available protein properties."""
    return {
        "properties": {
            "thermostable": "Resistance to high temperatures (0-1)",
            "enzyme": "Catalytic/enzymatic activity (0-1)",
            "membrane": "Membrane-associated properties (0-1)",
            "structural": "Structural stability (0-1)",
            "flexible": "Molecular flexibility (0-1)"
        }
    }

class ExportRequest(BaseModel):
    sequences: list

@app.post("/export")
def export_fasta(request: ExportRequest):
    """Export sequences as FASTA format."""
    fasta_lines = []
    
    for i, seq in enumerate(request.sequences):
        fasta_lines.append(f">Protogen_Sequence_{i+1}|Length={len(seq)}")
        for j in range(0, len(seq), 60):
            fasta_lines.append(seq[j:j+60])
    
    return {"fasta": "\n".join(fasta_lines)}

class StructureRequest(BaseModel):
    sequence: str
@app.post("/predict_structure")
async def predict_structure(request: StructureRequest):
    """
    Predict 3D structure using ESMFold API with retries.
    
    Args:
        request: Request containing protein sequence
        
    Returns:
        PDB structure data
    """
    try:
        import requests
        import time
        
        # Validate sequence length
        if len(request.sequence) > 400:
            return {"error": "Sequence too long (max 400 amino acids for API)", "success": False}
        
        # ESMFold API endpoint
        url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
        
        # Try up to 3 times with exponential backoff
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.post(url, data=request.sequence, timeout=120)
                
                if response.status_code == 200:
                    pdb_data = response.text
                    return {"pdb": pdb_data, "success": True}
                elif response.status_code == 503:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5  # Wait longer between retries
                        print(f"API busy, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return {"error": "ESMFold API is currently overloaded. Try a shorter sequence or wait a few minutes.", "success": False}
                else:
                    return {"error": f"API returned status {response.status_code}", "success": False}
            
            except requests.Timeout:
                return {"error": "Request timed out. Try a shorter sequence.", "success": False}
    
    except Exception as e:
        return {"error": str(e), "success": False}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)