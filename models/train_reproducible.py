"""Small CPU experiment with exact-sequence deduplication and an explicit causal objective."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader, Dataset
from models.conditional_transformer import ConditionalProteinTransformer
from data.scripts.tokenizer import AminoAcidTokenizer
from data.scripts.process_large_dataset import assign_properties

ROOT = Path(__file__).resolve().parents[1]
PROPERTIES = ["thermostable", "enzyme", "membrane", "structural", "flexible"]


def split_sequences(sequences, seed=42):
    unique = sorted({s.strip().upper() for s in sequences if s.strip()})
    if len(unique) < 10 or any(set(s) - set(AminoAcidTokenizer.AMINO_ACIDS) for s in unique):
        raise ValueError("Need at least 10 unique canonical amino-acid sequences")
    random.Random(seed).shuffle(unique)
    n = len(unique)
    a = int(n * 0.8)
    b = a + max(1, int(n * 0.1))
    return {"train": unique[:a], "validation": unique[a:b], "test": unique[b:]}


class SequenceDataset(Dataset):
    def __init__(self, sequences, max_length):
        self.sequences = sequences
        self.max_length = max_length
        self.tokenizer = AminoAcidTokenizer()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        # Proxies correspond to the retained residues, not discarded suffixes.
        seq = self.sequences[i][: self.max_length - 2]
        props = assign_properties(seq)
        tokens = self.tokenizer.pad_sequence(self.tokenizer.encode(seq), self.max_length)
        return torch.tensor(tokens), torch.tensor(
            [props[k] for k in PROPERTIES], dtype=torch.float32
        )


def evaluate(model, loader, baseline_log_probs):
    model.eval()
    loss_sum = baseline_sum = count = 0
    with torch.no_grad():
        for tokens, props in loader:
            targets = tokens[:, 1:]
            valid = targets.ne(20)
            logits = model(tokens[:, :-1], properties=props)
            loss_sum += torch.nn.functional.cross_entropy(
                logits.reshape(-1, 24),
                targets.reshape(-1),
                ignore_index=20,
                reduction="sum",
            ).item()
            baseline_sum += -baseline_log_probs[targets[valid]].sum().item()
            count += valid.sum().item()
    return {
        "token_count": count,
        "nll": loss_sum / count,
        "perplexity": math.exp(loss_sum / count),
        "unigram_nll": baseline_sum / count,
        "unigram_perplexity": math.exp(baseline_sum / count),
    }


def run(data_path, output, epochs=3, seed=42, max_length=64):
    if not 1 <= epochs <= 100 or not 8 <= max_length <= 512:
        raise ValueError("Invalid epoch/length bounds")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    path = Path(data_path)
    raw = json.loads(path.read_text())["sequences"]
    assert all(isinstance(s, str) for s in raw), "Use raw sequence strings"
    splits = split_sequences(raw, seed)
    retained = {key: {s[: max_length - 2] for s in seqs} for key, seqs in splits.items()}
    keys = list(retained)
    if any(retained[a] & retained[b] for i, a in enumerate(keys) for b in keys[:i]):
        raise ValueError(
            "Truncation creates cross-split duplicates; increase length or group matching prefixes"
        )
    datasets = {k: SequenceDataset(v, max_length) for k, v in splits.items()}
    loaders = {
        k: DataLoader(
            v,
            batch_size=8,
            shuffle=k == "train",
            generator=torch.Generator().manual_seed(seed),
        )
        for k, v in datasets.items()
    }
    config = dict(
        vocab_size=24,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        max_seq_length=512,
        num_properties=5,
        dropout=0.1,
    )
    model = ConditionalProteinTransformer(**config)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    counts = torch.ones(24)
    for tokens, _ in datasets["train"]:
        targets = tokens[1:]
        counts += torch.bincount(targets[targets.ne(20)], minlength=24)
    counts[20:23] = 0
    baseline_log_probs = (counts / counts.sum()).log()
    best = float("inf")
    best_weights = None
    history = []
    for epoch in range(epochs):
        model.train()
        for tokens, props in loaders["train"]:
            opt.zero_grad()
            logits = model(tokens[:, :-1], properties=props)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 24), tokens[:, 1:].reshape(-1), ignore_index=20
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
        result = evaluate(model, loaders["validation"], baseline_log_probs)
        history.append({"epoch": epoch + 1, **result})
        if result["nll"] < best:
            best = result["nll"]
            best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_weights is not None, "Training must select validation weights"
    model.load_state_dict(best_weights)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    AminoAcidTokenizer().save(output / "tokenizer.json")
    ids = {k: [hashlib.sha256(s.encode()).hexdigest() for s in v] for k, v in splits.items()}
    report = {
        "protocol": "causal-exact-dedup-v1",
        "seed": seed,
        "epochs": epochs,
        "model_config": config,
        "torch_version": str(torch.__version__),
        "device": "cpu",
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256((output / "model.pt").read_bytes()).hexdigest(),
        "max_training_tokens": max_length,
        "raw_rows": len(raw),
        "unique_sequences": sum(map(len, splits.values())),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "split_ids": ids,
        "history": history,
        "test": evaluate(model, loaders["test"], baseline_log_probs),
        "limitations": [
            "Exact deduplication only; no homology-disjoint split.",
            "Sequences truncated to configured length; five composition proxies are not measured biological properties.",
            "Tiny pipeline experiment, not evidence of functional proteins or conditioning efficacy.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "raw_rows",
                    "unique_sequences",
                    "split_counts",
                    "test",
                    "limitations",
                ]
            },
            indent=2,
        )
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/large_dataset.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/latest")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=64)
    args = parser.parse_args()
    run(args.data, args.output, args.epochs, args.seed, args.max_length)


if __name__ == "__main__":
    main()
