"""Check whether the fixed Dress and Ankle boot classes have credible GAN endpoints."""
import argparse
import json
from pathlib import Path

import torch

from models import Adversary, Classifier
from render_final_figures import show_rows


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", type=Path, default=Path("../FINAL_CHECKPOINTS"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    a.out.mkdir(parents=True, exist_ok=True)
    clf = Classifier().eval()
    clf.load_state_dict(torch.load(a.checkpoints/"classifier_full"/"MLP.pt",
                                   map_location="cpu", weights_only=True)["state"])
    result = {"seed": 42, "target_labels": [3, 9], "target_names": ["Dress", "Ankle boot"],
              "candidates_per_model": 4096, "labels_are_predictions": True, "models": {}}
    rows, titles = [], []
    for name in ("GAN", "DCGAN"):
        m = Adversary(name == "DCGAN").eval()
        m.load_state_dict(torch.load(a.checkpoints/"full_budget"/f"{name}.pt",
                                     map_location="cpu", weights_only=True)["state"])
        z = torch.randn(4096, m.z_dim, generator=torch.Generator().manual_seed(42))
        generated = torch.cat([m.g(chunk) for chunk in z.split(128)])
        confidence, pred = clf(generated).softmax(1).max(1)
        stats = {}
        previews = []
        for label, title in zip(result["target_labels"], result["target_names"]):
            matches = (pred == label).nonzero().flatten()
            best = matches[confidence[matches].argmax()] if len(matches) else None
            stats[title] = {"predicted_count": len(matches),
                            "largest_softmax": float(confidence[best]) if best is not None else None}
            previews.append(generated[best] if best is not None else torch.zeros_like(generated[0]))
        result["models"][name] = stats
        rows.append(torch.stack(previews))
        titles.append(name)
    show_rows(a.out/"same_class_candidates.png", rows, titles,
              ["Prediksi Dress", "Prediksi Ankle boot"])
    (a.out/"same_class_candidates.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
