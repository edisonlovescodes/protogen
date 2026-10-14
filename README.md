# Protogen

A small PyTorch amino-acid sequence modeling project with a FastAPI generation API, React interface and FASTA export. The model accepts five **heuristic composition proxy scores**. These are not experimentally measured thermostability, catalytic activity or other biological properties.

## October 2026 repair

The original training objective lacked a causal attention mask, and the larger dataset had 425 rows containing only 85 unique sequences. The refreshed pipeline:

- Deduplicates exact sequence strings **before** a seeded train/validation/test split.
- Always blocks future tokens and padded keys in attention.
- Retains START/END tokens when truncating and uses one training entry point.
- Records token-weighted held-out NLL/perplexity and a training-only unigram baseline.
- Saves a checksum-verified model/config/tokenizer/split/report bundle.
- Bounds generation inputs; fails closed without the new bundle; uses root-independent paths.

Historical checkpoints remain in Git history/current model files for comparison, but the API does not silently load them. Old conditioning/throughput percentages and the old noncausal validation loss are not valid new evaluation results.

## Reproduce the small CPU experiment

Python 3.12, from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m models.train_reproducible --epochs 3 --max-length 64
```

This uses the existing raw dataset and writes `runs/latest/{model.pt,tokenizer.json,report.json}`. No data download, GPU or cloud service is required. The small experiment checks the repaired pipeline; its output is not proof of functional proteins. Raw-sequence checksums and split IDs are recorded. Exact deduplication does **not** establish a homology-disjoint split; related sequences can remain across partitions. Conditioning adherence needs a separate paired-seed/control experiment.

The default run retains at most 62 amino acids plus START/END for training. The positional buffer supports 512 tokens, but API generation is capped at 150 tokens, temperature (0,2], and 1–10 sequences. Longer generation than the training window is exploratory.

## Run the app

```sh
uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

In another terminal:

```sh
cd frontend
npm ci --ignore-scripts
npm run dev
```

Open http://localhost:5173. Vite routes `/api` to the local backend; `VITE_API_BASE_URL` can override the URL at build time. API documentation: http://127.0.0.1:8000/docs. `PROTOGEN_MODEL_DIR` selects a different compatible bundle. `CORS_ORIGINS` configures explicit separate-host origins.

External folding and the old unverified viewer link have been removed from the interface; the old route returns 503. No external service is contacted during startup, tests or generation.

## Tests

```sh
pip install -r requirements-dev.txt
python -m pytest tests -q
cd frontend
npm run build
```

Tests cover causal-prefix invariance, padding, deduplication, truncation/EOS, a tiny train/save/load workflow, input rejection, checksum checks, missing-model behavior and FASTA validation. CI runs the regression suite and frontend build. Old `models/train*.py` entry points forward to the same reproducible protocol.

See [CLAIMS.md](CLAIMS.md) and [CHANGELOG.md](CHANGELOG.md). Generated dependencies, bytecode, build output and local runs are ignored; package locks remain tracked.

## Interface and type

The interface uses self-hosted typefaces distributed by [Google Fonts](https://github.com/google/fonts). Original SIL Open Font License files are included in `frontend/public/fonts/`. No hosted font request is needed to render the page.

Frontend regression tests: `cd frontend` then `npm test`. `PROTOGEN_API_PROXY` can configure the Vite development proxy target for parallel local previews.
