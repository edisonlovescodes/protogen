# 🧬 Protogen - AI-Powered Protein Sequence Generator

An end-to-end machine learning application that generates novel protein sequences with specific biochemical properties using a Transformer-based neural network.

![Protogen Demo](https://img.shields.io/badge/Status-Live-success) ![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange)

## 🌟 Features

- **Property-Conditioned Generation**: Generate proteins with specific traits (thermostability, enzyme activity, membrane association, etc.)
- **Real-Time Prediction**: Interactive web interface for instant protein generation
- **Batch Generation**: Create multiple sequences simultaneously
- **FASTA Export**: Download sequences in standard bioinformatics format
- **3D Structure Prediction**: Integration with ESMFold API for structure visualization
- **Large-Scale Training**: Trained on 425+ real protein sequences from UniProt

## 🏗️ Architecture

### Model
- **Type**: Conditional Transformer
- **Parameters**: 3.17 million
- **Architecture**: 4 layers, 256 hidden dimensions, 8 attention heads
- **Training**: 10 epochs on curated UniProt dataset
- **Properties**: 5-dimensional conditioning (thermostable, enzyme, membrane, structural, flexible)

### Tech Stack
**Backend:**
- FastAPI (REST API)
- PyTorch (Deep Learning)
- Keras (Model training)

**Frontend:**
- React
- Vite
- Pure CSS (no frameworks)

**Data:**
- UniProt API (protein sequences)
- Custom tokenizer for amino acid encoding

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- 4GB+ RAM

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/edisonlovescodes/protogen.git
cd protogen

Set up backend

bashpython3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Set up frontend

bashcd frontend
npm install

Run the application

Terminal 1 (Backend):
bashcd protogen
python backend/api.py
Terminal 2 (Frontend):
bashcd protogen/frontend
npm run dev

Open browser

http://localhost:3000
📊 Model Training
The model was trained through several phases:

Data Collection: Downloaded 425 protein sequences from UniProt
Preprocessing: Tokenized sequences and computed biochemical properties
Training: 10 epochs with Adam optimizer, learning rate 5e-4
Validation: 85/15 train/val split, achieving loss of 2.81

🎯 API Endpoints

POST /generate - Generate protein sequences
POST /export - Export as FASTA format
POST /predict_structure - Predict 3D structure (ESMFold)
GET /properties - List available properties
GET /health - Health check

📁 Project Structure
protogen/
├── backend/
│   └── api.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── main.jsx
│   └── package.json
├── models/
│   ├── transformer.py
│   ├── conditional_transformer.py
│   ├── train_improved.py
│   └── best_model.pt
├── data/
│   ├── scripts/
│   └── processed/
└── requirements.txt
🧪 Example Usage
Generate a thermostable protein:
pythonimport requests

response = requests.post('http://localhost:8000/generate', json={
    'max_length': 100,
    'temperature': 1.0,
    'num_sequences': 1,
    'properties': {
        'thermostable': 0.9,
        'enzyme': 0.2,
        'membrane': 0.1,
        'structural': 0.5,
        'flexible': 0.3
    }
})

print(response.json()['sequences'])
📈 Results

Successfully generates diverse protein sequences
Property conditioning demonstrably affects amino acid composition
Model converges with loss < 3.0
Generated sequences pass basic biochemical validation

## 📈 Future Enhancements
- [ ] Multi-protein complex generation
- [ ] Integration with AlphaFold 3 for improved folding
- [ ] Sequence alignment and homology analysis
- [ ] Fine-tuning on specific protein families (enzymes, antibodies)
- [ ] Protein function prediction from sequence
- [ ] Advanced property conditioning (pH stability, solubility)
- [ ] User authentication & project management
- [ ] PostgreSQL for production database
- [ ] Real-time collaboration features
- [ ] Export to multiple formats (PDB, mmCIF, FASTA)

## 👨‍💻 Author

**Edison Essien**
- Portfolio: [edisoncodes.ai](https://edisoncodes.ai)
- LinkedIn: [@essienedison](https://linkedin.com/in/essienedison)
- GitHub: [edisonlovescodes](https://github.com/edisonlovescodes)
- Email: organicedison@gmail.com

## 🙏 Acknowledgments

- **UniProt Database**: The Universal Protein Resource
- **ESMFold**: Meta AI - Evolutionary Scale Modeling
- **PyTorch**: Facebook AI Research
- **FastAPI**: Sebastián Ramírez
- **React**: Facebook/Meta
- **Transformers Architecture**: Vaswani et al.

---

Built with ❤️ and lots of ☕
