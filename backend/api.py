"""Bounded generation from a versioned causal experiment bundle."""

from contextlib import asynccontextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Annotated
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
import torch
from models.conditional_transformer import ConditionalProteinTransformer
from data.scripts.tokenizer import AminoAcidTokenizer

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["thermostable", "enzyme", "membrane", "structural", "flexible"]
LOGGER = logging.getLogger(__name__)
Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Properties(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thermostable: Score = 0.5
    enzyme: Score = 0.5
    membrane: Score = 0.5
    structural: Score = 0.5
    flexible: Score = 0.5


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_length: int = Field(100, ge=2, le=150, strict=True)
    temperature: float = Field(1, gt=0, le=2, allow_inf_nan=False)
    num_sequences: int = Field(1, ge=1, le=10, strict=True)
    properties: Properties = Field(default_factory=Properties)


Sequence = Annotated[str, Field(min_length=1, max_length=512, pattern=r"^[ARNDCQEGHILKMFPSTWYV]+$")]


class ExportRequest(BaseModel):
    sequences: list[Sequence] = Field(min_length=1, max_length=10)


def create_app(bundle_path=None):
    bundle = Path(bundle_path or os.getenv("PROTOGEN_MODEL_DIR", ROOT / "runs/latest")).resolve()
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.model = None
        app.state.report = None
        app.state.tokenizer = None
        try:
            report = json.loads((bundle / "report.json").read_text())
            if report["protocol"] != "causal-exact-dedup-v1":
                raise ValueError(
                    "Unsupported model protocol; historical checkpoints need retraining"
                )
            digest = hashlib.sha256((bundle / "model.pt").read_bytes()).hexdigest()
            if digest != report["model_sha256"]:
                raise ValueError("Artifact checksum mismatch")
            tokenizer = AminoAcidTokenizer.load(bundle / "tokenizer.json")
            if tokenizer.aa_to_token != AminoAcidTokenizer().aa_to_token:
                raise ValueError("Tokenizer does not match this model contract")
            model = ConditionalProteinTransformer(**report["model_config"])
            model.load_state_dict(
                torch.load(bundle / "model.pt", map_location="cpu", weights_only=True), strict=True
            )
            model.eval()
            app.state.model = model
            app.state.tokenizer = tokenizer
            app.state.report = report
        except Exception:
            LOGGER.exception("No usable causal bundle; generation will return 503")
        yield
        app.state.model = None

    app = FastAPI(title="Protogen API", version="3.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            x.strip()
            for x in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
            if x.strip()
        ],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/")
    def root():
        return {
            "name": "Protogen",
            "version": "3.1.0",
            "scope": "Amino-acid sequence prototype conditioned on composition proxies",
        }

    @app.get("/health")
    def health():
        ready = app.state.model is not None
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ready" if ready else "model_unavailable",
                "model_loaded": ready,
                "protocol": app.state.report["protocol"] if ready else None,
                "model_sha256": app.state.report["model_sha256"] if ready else None,
            },
        )

    @app.post("/generate")
    def generate(request: GenerateRequest):
        if app.state.model is None:
            raise HTTPException(
                503, "Run the documented local experiment to create a causal model bundle"
            )
        if request.max_length > app.state.model.pos_encoding.pe.shape[1]:
            raise HTTPException(422, "Length exceeds model capacity")
        props = request.properties.model_dump()
        sequences = []
        tokens = []
        try:
            with lock:
                for _ in range(request.num_sequences):
                    generated = app.state.model.generate_with_properties(
                        [props[k] for k in KEYS],
                        max_length=request.max_length,
                        temperature=request.temperature,
                        device="cpu",
                    )
                    sequences.append(app.state.tokenizer.decode(generated))
                    tokens.append(generated)
        except Exception:
            LOGGER.exception("Generation failed")
            raise HTTPException(500, "Sequence generation failed") from None
        return {
            "sequences": sequences,
            "tokens": tokens,
            "properties": props,
            "protocol": app.state.report["protocol"],
            "model_sha256": app.state.report["model_sha256"],
            "note": "Composition proxy controls; biological function and conditioning efficacy are unvalidated",
        }

    @app.get("/properties")
    def properties():
        return {
            "properties": {
                k: "Heuristic composition proxy in [0,1]; not measured biological function"
                for k in KEYS
            }
        }

    @app.post("/export")
    def export(request: ExportRequest):
        lines = []
        for i, seq in enumerate(request.sequences, 1):
            lines.append(f">Protogen_Sequence_{i}|Length={len(seq)}")
            lines.extend(seq[j : j + 60] for j in range(0, len(seq), 60))
        return {"fasta": "\n".join(lines) + "\n"}

    @app.post("/predict_structure")
    def structure():
        raise HTTPException(503, "External folding is not enabled in this local prototype")

    return app


app = create_app()
