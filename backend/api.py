"""Bounded generation from a versioned causal experiment bundle."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
import hashlib
import json
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


@dataclass(frozen=True)
class LoadedBundle:
    model: ConditionalProteinTransformer
    tokenizer: AminoAcidTokenizer
    report: dict


def create_app(bundle: Path):
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app):
        app.state.bundle = None
        if bundle.exists():
            report = json.loads((bundle / "report.json").read_text())
            assert report["protocol"] == "causal-exact-dedup-v1", "Unsupported model protocol"
            digest = hashlib.sha256((bundle / "model.pt").read_bytes()).hexdigest()
            assert digest == report["model_sha256"], "Artifact checksum mismatch"
            tokenizer = AminoAcidTokenizer.load(bundle / "tokenizer.json")
            model = ConditionalProteinTransformer(**report["model_config"])
            model.load_state_dict(
                torch.load(bundle / "model.pt", map_location="cpu", weights_only=True)
            )
            app.state.bundle = LoadedBundle(model.eval(), tokenizer, report)
        try:
            yield
        finally:
            app.state.bundle = None

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
        loaded = app.state.bundle
        ready = loaded is not None
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ready" if ready else "model_unavailable",
                "model_loaded": ready,
                "protocol": loaded.report["protocol"] if ready else None,
                "model_sha256": loaded.report["model_sha256"] if ready else None,
            },
        )

    @app.post("/generate")
    def generate(request: GenerateRequest):
        loaded = app.state.bundle
        if loaded is None:
            raise HTTPException(
                503,
                "Run the documented local experiment to create a causal model bundle",
            )
        if request.max_length > loaded.model.pos_encoding.pe.shape[1]:
            raise HTTPException(422, "Length exceeds model capacity")
        props = request.properties.model_dump()
        sequences = []
        tokens = []
        with lock:
            for _ in range(request.num_sequences):
                generated = loaded.model.generate_with_properties(
                    [props[k] for k in KEYS], request.max_length, request.temperature
                )
                sequences.append(loaded.tokenizer.decode(generated))
                tokens.append(generated)
        return {
            "sequences": sequences,
            "tokens": tokens,
            "properties": props,
            "protocol": loaded.report["protocol"],
            "model_sha256": loaded.report["model_sha256"],
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


app = create_app(Path(os.getenv("PROTOGEN_MODEL_DIR", ROOT / "runs/latest")))
