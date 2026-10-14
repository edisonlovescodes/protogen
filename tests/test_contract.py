import json
import pytest
import torch
from fastapi.testclient import TestClient
from models.transformer import ProteinTransformer
from models.conditional_transformer import ConditionalProteinTransformer
from models.train_reproducible import split_sequences, run
from data.scripts.tokenizer import AminoAcidTokenizer
from backend.api import create_app


@pytest.mark.parametrize("conditional", [False, True])
def test_future_residues_cannot_change_prefix_logits(conditional):
    torch.manual_seed(3)
    cls = ConditionalProteinTransformer if conditional else ProteinTransformer
    model = cls(vocab_size=24, d_model=16, num_layers=2, num_heads=2, d_ff=32, dropout=0).eval()
    original = torch.tensor([[22, 1, 2, 3, 4, 23, 20]])
    changed = original.clone()
    changed[:, 4:6] = torch.tensor([7, 9])
    kwargs = {"properties": torch.tensor([[0.5] * 5])} if conditional else {}
    with torch.no_grad():
        a = model(original, **kwargs)
        b = model(changed, **kwargs)
    torch.testing.assert_close(a[:, :4], b[:, :4], rtol=0, atol=1e-6)


def test_padding_does_not_change_prefix():
    model = ProteinTransformer(24, d_model=16, num_layers=1, num_heads=2, d_ff=32, dropout=0).eval()
    with torch.no_grad():
        a = model(torch.tensor([[22, 1, 23]]))
        b = model(torch.tensor([[22, 1, 23, 20, 20]]))
    torch.testing.assert_close(a, b[:, :3], atol=1e-6, rtol=0)


def test_deduplicate_before_split():
    seqs = ["A" * (i + 1) for i in range(20)] * 5
    splits = split_sequences(seqs, 42)
    groups = [set(v) for v in splits.values()]
    assert sum(map(len, groups)) == 20
    assert all(not groups[i] & groups[j] for i in range(3) for j in range(i))
    assert splits == split_sequences(seqs, 42)


def test_truncation_keeps_eos():
    tokenizer = AminoAcidTokenizer()
    assert tokenizer.pad_sequence(tokenizer.encode("A" * 100), 8) == [22] + [0] * 6 + [23]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("causal")
    data = root / "data.json"
    data.write_text(json.dumps({"sequences": ["ARND" + "A" * i for i in range(1, 21)] * 2}))
    report = run(data, root / "bundle", epochs=1, max_length=16)
    assert report["unique_sequences"] == 20
    assert report["test"]["token_count"] > 0
    return root / "bundle"


@pytest.fixture
def client(bundle):
    with TestClient(create_app(bundle)) as c:
        yield c


@pytest.mark.parametrize(
    "payload",
    [
        {"temperature": 0},
        {"num_sequences": -1},
        {"num_sequences": 11},
        {"max_length": 151},
        {"properties": {"enzyme": 1.1}},
        {"properties": {"unknown": 0.5}},
    ],
)
def test_invalid_generation_is_rejected(client, payload):
    assert client.post("/generate", json=payload).status_code == 422


def test_bundle_generation_and_export(client):
    assert client.get("/health").json()["model_loaded"] is True
    r = client.post("/generate", json={"max_length": 16, "num_sequences": 2})
    assert r.status_code == 200
    b = r.json()
    assert len(b["sequences"]) == 2
    assert all(set(s) <= set(AminoAcidTokenizer.AMINO_ACIDS) for s in b["sequences"])
    assert all(20 not in t and 21 not in t and 22 not in t[1:] for t in b["tokens"])
    # An empty generated sequence is possible (immediate EOS); only nonempty sequences are exportable.
    fasta = client.post("/export", json={"sequences": ["ARND", "MKT"]})
    assert fasta.status_code == 200
    assert ">Protogen_Sequence_1|Length=4\nARND" in fasta.json()["fasta"]
    assert client.post("/export", json={"sequences": ["AA\n>injected"]}).status_code == 422


def test_missing_bundle_and_disabled_folding(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/generate", json={}).status_code == 503
        assert c.post("/predict_structure", json={"sequence": "ARND"}).status_code == 503


def test_changed_artifact_fails_closed(bundle, tmp_path):
    import shutil

    shutil.copytree(bundle, tmp_path / "bundle")
    with (tmp_path / "bundle/model.pt").open("ab") as f:
        f.write(b"changed")
    with TestClient(create_app(tmp_path / "bundle")) as c:
        assert c.get("/health").status_code == 503


def test_truncation_collision_is_rejected(tmp_path):
    path = tmp_path / "prefixes.json"
    path.write_text(json.dumps({"sequences": ["A" * 20 + "C" * i for i in range(1, 21)]}))
    with pytest.raises(ValueError, match="cross-split duplicates"):
        run(path, tmp_path / "bundle", epochs=1, max_length=8)
