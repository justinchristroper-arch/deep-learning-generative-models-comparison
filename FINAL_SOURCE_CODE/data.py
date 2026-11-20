import gzip
import hashlib
import os
import struct
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset


FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}


def read_idx(root, name):
    p = root / name
    if not p.exists():
        url = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/" + name
        tmp = p.with_suffix(".partial")
        urllib.request.urlretrieve(url, tmp)
        if hashlib.md5(tmp.read_bytes()).hexdigest() != FILES[name]:
            raise ValueError(f"Download checksum mismatch for {name}")
        os.replace(tmp, p)
    if hashlib.md5(p.read_bytes()).hexdigest() != FILES[name]:
        raise ValueError(f"Checksum mismatch for {p}")
    with gzip.open(p, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        dims = (n, 1, *struct.unpack(">II", f.read(8))) if magic == 2051 else (n,)
        return torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).reshape(dims)


class Images(TensorDataset):
    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        return x.float()/255, y.long()


def loaders(root, out, mode, seed=42):
    g = torch.Generator().manual_seed(seed)
    if mode == "smoke":
        ds = Images(torch.randint(256, (320, 1, 28, 28), generator=g, dtype=torch.uint8),
                    torch.randint(10, (320,), generator=g))
    else:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        try:
            from torchvision.datasets import FashionMNIST
        except (ImportError, RuntimeError, OSError):
            FashionMNIST = None
        if FashionMNIST and not all((root/k).exists() for k in FILES):
            train, test = [FashionMNIST(root, train=v, download=True) for v in (True, False)]
            ds = Images(torch.cat([train.data, test.data]).unsqueeze(1),
                        torch.cat([train.targets, test.targets]))
        else:
            parts = [read_idx(root, k) for k in FILES]
            ds = Images(torch.cat(parts[::2]), torch.cat(parts[1::2]))
    perm = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))
    a, b = len(ds)*8//10, len(ds)*9//10
    splits = dict(train=perm[:a], validation=perm[a:b], test=perm[b:])
    if mode == "quick":
        torch.save(splits, out/"full_indices.pt")
        for key, count in zip(splits, [4096, 512, 512]):
            splits[key] = splits[key][:count]
    index_path=out/"indices.pt"
    if index_path.exists():
        prior=torch.load(index_path,weights_only=True,map_location="cpu")
        if set(prior)!=set(splits) or any(not torch.equal(splits[k],prior[k]) for k in splits):
            raise ValueError("Saved split membership differs from the reconstructed seed-42 experiment")
    else:
        torch.save(splits,index_path)
    return {k: DataLoader(Subset(ds, ix.tolist()), batch_size=128, shuffle=k == "train",
                          generator=torch.Generator().manual_seed(seed)) for k, ix in splits.items()}
