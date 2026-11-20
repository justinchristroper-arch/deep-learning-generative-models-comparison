"""Paired, synthetic defect stress test on held-out Fashion-MNIST."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import rankdata

from data import loaders
from models import AE

KINDS = ("Occlusion", "Speckle", "Local shift")


def defects(images, seed):
    """Apply one of three equally allocated defects inside a foreground region."""
    rng = torch.Generator().manual_seed(seed)
    output = images.clone()
    kinds = torch.empty(len(images), dtype=torch.long)
    kinds[torch.randperm(len(images), generator=rng)] = torch.arange(len(images)) % len(KINDS)
    for i, image in enumerate(images):
        centers = (image[0, 3:25, 3:25] > 0.35).nonzero()
        if not len(centers):
            centers = (image[0, 3:25, 3:25] > image[0, 3:25, 3:25].max() * 0.5).nonzero()
        center = centers[torch.randint(len(centers), (), generator=rng)] + 3 if len(centers) else (14, 14)
        row, col = map(int, center)
        patch = image[:, row-3:row+4, col-3:col+4]
        if kinds[i] == 0:
            changed = patch * 0.15
        elif kinds[i] == 1:
            signs = torch.randint(2, patch.shape, generator=rng) * 2 - 1
            changed = (patch + 0.45 * signs).clamp(0, 1)
        else:
            changed = torch.zeros_like(patch)
            changed[..., 2:] = patch[..., :-2]
        output[i, :, row-3:row+4, col-3:col+4] = changed
    return output, kinds


def comparison(normal, anomalous, validated_normal):
    """Tie-aware rank AUROC, threshold-grouped average precision and ROC TPR."""
    scores = np.concatenate([normal, anomalous])
    labels = np.arange(len(scores)) >= len(normal)
    order = np.argsort(-scores, kind="stable")
    end = np.r_[np.flatnonzero(np.diff(scores[order])), len(scores) - 1]
    tp = np.cumsum(labels[order])[end]
    fp = end + 1 - tp
    recall = tp / len(anomalous)
    precision = tp / (end + 1)
    permitted = validated_normal.size // 20
    threshold = np.sort(validated_normal)[-permitted-1]
    return {
        "auroc": float((rankdata(scores)[labels].sum() - len(anomalous) * (len(anomalous)+1)/2)
                       / (len(normal) * len(anomalous))),
        "auprc_average_precision": float(np.dot(np.diff(np.r_[0, recall]), precision)),
        "test_tpr_at_test_fpr_at_most_5pct": float(np.max(recall[20 * fp <= len(normal)], initial=0)),
        "validation_threshold": float(threshold),
        "validation_fpr_at_threshold": float(np.mean(validated_normal > threshold)),
        "test_fpr_at_validation_threshold": float(np.mean(normal > threshold)),
        "test_tpr_at_validation_threshold": float(np.mean(anomalous > threshold)),
        "median_normal_mse": float(np.median(normal)),
        "median_anomaly_mse": float(np.median(anomalous)),
    }


@torch.inference_mode()
def reconstruction_errors(model, images):
    return torch.cat([(model(x)[0] - x).square().mean((1, 2, 3))
                      for x in images.split(128)]).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, default=Path("../FINAL_CHECKPOINTS/full_budget"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    args.out.mkdir(parents=True, exist_ok=True)
    split = loaders(args.data, args.out, "full")
    indices = torch.load(args.out / "indices.pt", weights_only=True)
    original = torch.load(args.checkpoints / "indices.pt", weights_only=True)
    assert all(torch.equal(indices[key], original[key]) for key in indices)
    expected = torch.randperm(70000, generator=torch.Generator().manual_seed(42))
    assert torch.equal(torch.cat(list(indices.values())), expected)
    validation = torch.cat([x for x, _ in split["validation"]])
    test = torch.cat([x for x, _ in split["test"]])
    anomalous, kind = defects(test, 42)
    result = {
        "seed": 42, "validation_normal_count": len(validation),
        "test_normal_count": len(test), "test_anomaly_count": len(anomalous),
        "test_counts_by_type": dict(zip(KINDS, torch.bincount(kind).tolist())),
        "score": "Mean pixelwise squared reconstruction error on the given input",
        "defects": "7x7 foreground patch: intensity *0.15, signed speckle ±0.45, or 2px right shift with zero-fill",
        "threshold_protocol": "Strictly above threshold; calibrated at 5% maximum FPR on clean validation only.",
        "test_roc_protocol": "Largest empirical TPR among grouped test ROC points with FPR <= 5%; retrospective, not a calibrated detector.",
        "plot_display": "Histogram display truncated at MSE 0.07. Metrics use all observations.",
        "models": {},
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.4), layout="constrained", sharey=True)
    for ax, name in zip(axes, ("AE", "SparseAE")):
        path = args.checkpoints / f"{name}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = AE().eval()
        model.load_state_dict(checkpoint["state"], strict=True)
        val, clean, corrupt = (reconstruction_errors(model, x) for x in (validation, test, anomalous))
        metrics = comparison(clean, corrupt, val)
        result["models"][name] = dict(
            metrics, epoch=checkpoint["epoch"], checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            by_type={label: comparison(clean[kind == i], corrupt[kind == i], val)
                     for i, label in enumerate(KINDS)},
        )
        edges = np.linspace(0, 0.07, 40)
        ax.hist(clean, bins=edges, histtype="step", density=True, label="Normal")
        ax.hist(corrupt, bins=edges, histtype="step", density=True, label="Cacat sintetis")
        ax.axvline(metrics["validation_threshold"], ls="--", lw=1, color="#555555", label="Ambang validasi")
        ax.set(title=name, xlabel="MSE rekonstruksi")
    axes[0].set_ylabel("Kepadatan")
    axes[1].legend(fontsize=7)
    fig.savefig(args.out / "anomaly_score_distribution.png", dpi=180)
    plt.close(fig)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
