"""Compatibility entry point; use the reproducible causal CPU experiment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.train_reproducible import main

if __name__ == "__main__":
    main()
