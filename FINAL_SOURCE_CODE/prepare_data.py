"""Fetch the four official Fashion-MNIST files and record the fixed split."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from data import FILES, loaders, read_idx
from full_benchmark import split_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs/dataset"))
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        read_idx(args.data, name)
    split = loaders(args.data, args.out, "full")
    indices = torch.load(args.out / "indices.pt", weights_only=True)
    joined = torch.cat(list(indices.values()))
    if not torch.equal(joined.sort().values, torch.arange(70_000)):
        raise ValueError("Expected disjoint, exhaustive Fashion-MNIST indices")
    record = {
        "dataset": "Fashion-MNIST",
        "seed": 42,
        "counts": {key: len(value) for key, value in indices.items()},
        "split_sha256": split_fingerprint(split),
        "official_gzip_sha256": {
            name: hashlib.sha256((args.data / name).read_bytes()).hexdigest()
            for name in FILES
        },
    }
    (args.out / "dataset.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
