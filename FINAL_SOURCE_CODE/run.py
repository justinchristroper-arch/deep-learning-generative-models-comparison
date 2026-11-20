"""Train and evaluate the complete take-home experiment."""
import argparse
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data import loaders
from models import AE, Adversary, Classifier, DDPM, decoder


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def weight_hash(model):
    h = hashlib.sha256()
    for value in model.state_dict().values():
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def losses(name, m, x, y, clf, optimizers=None):
    if name in {"GAN", "DCGAN"}:
        z = torch.randn(len(x), m.z_dim, device=x.device)
        fake = m.g(z)
        pos, neg = m.d(x), m.d(fake.detach())
        dloss = (F.softplus(-pos).mean() + F.softplus(neg).mean())/2
        if optimizers:
            optimizers[1].zero_grad()
            dloss.backward()
            optimizers[1].step()
        gloss = F.softplus(-m.d(fake)).mean()
        return gloss, {"discriminator": dloss.item()}
    if name == "MLP":
        logits = m(x)
        return F.cross_entropy(logits, y), {"accuracy": (logits.argmax(1) == y).float().mean().item()}
    if name == "LatentDecoder":
        return F.mse_loss(m(clf.encode(x).detach()), x), {}
    if name == "DDPM":
        return m.loss(x), {}
    pred, z, lv = m(x)
    err = F.mse_loss(pred, x)
    if name == "SparseAE":
        return err + .001*z.abs().mean(), {"mse": err.item()}
    if name == "VAE":
        kl = -.5*(1 + lv - z.square() - lv.exp()).sum(1).mean()
        bce = F.binary_cross_entropy(pred, x, reduction="sum")/len(x)
        return (bce + kl)/784, {"mse": err.item(), "kl": kl.item()}
    return err, {}


def epoch(name, model, loader, clf, device, opts=None):
    model.train(bool(opts))
    total, count = {}, 0
    with torch.set_grad_enabled(bool(opts)):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss, values = losses(name, model, x, y, clf, opts)
            if opts:
                opts[0].zero_grad()
                loss.backward()
                opts[0].step()
            values["loss"] = loss.item()
            if not all(math.isfinite(v) for v in values.values()):
                raise FloatingPointError(f"Nonfinite {name} loss")
            for k, v in values.items():
                total[k] = total.get(k, 0.) + v*len(x)
            count += len(x)
    return {k: v/count for k, v in total.items()}


def main(adversary_factory=Adversary, fit_only=False):
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    for key in ("smoke", "quick", "full"):
        group.add_argument("--"+key, action="store_true")
    p.add_argument("--epochs", type=int)
    p.add_argument("--classifier-only", action="store_true", help="Run the classifier accuracy experiment alone")
    p.add_argument("--data", default="data")
    p.add_argument("--out", type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--fid-count", type=int, help="Real and fake samples per model (default 7000 full, 0 diagnostic)")
    args = p.parse_args()
    mode = "smoke" if args.smoke else "quick" if args.quick else "full"
    epochs = args.epochs if args.epochs is not None else {"full":30, "quick":3, "smoke":1}[mode]
    if epochs < 1:
        p.error("--epochs must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    out = args.out or Path("runs")/mode
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        p.error(f"Choose an empty output directory to preserve previous results: {out}")
    dl = loaders(args.data, out, mode)
    steps = 8 if mode == "smoke" else 1000
    count = args.fid_count if args.fid_count is not None else (7000 if mode == "full" else 0)
    if count != 0 and not 2 <= count <= len(dl["test"].dataset):
        p.error("--fid-count must be zero or between 2 and the test-set size")
    models = dict(MLP=Classifier(), AE=AE(), SparseAE=AE(), VAE=AE(True), GAN=adversary_factory(False),
                  DCGAN=adversary_factory(True), DDPM=DDPM(steps), LatentDecoder=decoder())
    if args.classifier_only:
        models = {"MLP": models["MLP"]}
    clf = models["MLP"]
    manifest = dict(mode=mode, dataset="synthetic random noise" if mode=="smoke" else "Fashion-MNIST",
                    seed=42, batch_size=128, epochs=epochs, ddpm_steps=steps, device=str(device),
                    hardware=torch.cuda.get_device_name(device) if device.type=="cuda" else platform.processor(),
                    platform=platform.platform(), python=platform.python_version(), torch=str(torch.__version__),
                    counts={k:len(v.dataset) for k,v in dl.items()}, status="training",
                    classifier_hidden=[256,64], latent_dim=64, dcgan_noise_dim=100,
                    optimizer="Adam", lr=.001, adversarial_lr=.0002, adversarial_betas=[.5,.999],
                    sparse_l1_weight=.001, vae_kl_weight=1, ddpm_beta_range=[.0001,.02],
                    evaluation_seed=42, requested_fid_count=count if not args.classifier_only else 0,
                    classifier_only=args.classifier_only, cpu_threads=args.threads)
    dump(out/"manifest.json", manifest)
    history = []
    for name, model in models.items():
        model.to(device)
        if name == "LatentDecoder":
            before = weight_hash(clf)
        adv = name in {"GAN", "DCGAN"}
        opts = [torch.optim.Adam(net.parameters(), lr=.0002, betas=(.5,.999)) for net in
                (model.g, model.d)] if adv else [torch.optim.Adam(model.parameters(), lr=.001)]
        best = math.inf
        for e in range(1, epochs+1):
            sync(device)
            start = time.perf_counter()
            train = epoch(name, model, dl["train"], clf, device, opts)
            sync(device)
            seconds = time.perf_counter()-start
            val = epoch(name, model, dl["validation"], clf, device)
            row = dict(model=name, epoch=e, seconds=seconds, train=train, validation=val)
            history.append(row)
            dump(out/"history.json", history)
            print(f"{name} {e}/{epochs} train={train['loss']:.4f} val={val['loss']:.4f}", flush=True)
            if adv or val["loss"] < best:
                best = val["loss"]
                torch.save(dict(state=model.state_dict(), epoch=e), out/(name+".pt"))
        model.load_state_dict(torch.load(out/(name+".pt"), map_location=device, weights_only=True)["state"])
        model.eval().requires_grad_(False)
        if name == "LatentDecoder":
            after = weight_hash(clf)
            if after != before:
                raise RuntimeError("Frozen classifier parameters changed during decoder training")
            manifest["classifier_sha256_before_decoder"] = before
            manifest["classifier_sha256_after_decoder"] = after
    if fit_only:
        manifest["status"] = "initial_training_finished_pending_dcgan_revision"
        manifest["discarded_dcgan_prototype"] = True
        dump(out/"manifest.json", manifest)
        return
    if args.classifier_only:
        test = epoch("MLP", clf, dl["test"], clf, device)
        outcome = dict(test_accuracy=test["accuracy"], test_cross_entropy=test["loss"],
                       accuracy_target=.88, target_met=test["accuracy"]>=.88,
                       selected_epoch=torch.load(out/"MLP.pt",weights_only=True,map_location="cpu")["epoch"])
        dump(out/"classifier_metrics.json",outcome)
    else:
        from evaluate import evaluate
        evaluate(models, dl, device, out, history, count)
    manifest["status"] = "complete"
    dump(out/"manifest.json", manifest)


if __name__ == "__main__":
    main()
