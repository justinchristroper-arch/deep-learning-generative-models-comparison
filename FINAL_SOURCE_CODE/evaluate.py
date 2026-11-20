"""Held-out metrics, timings and five comparison figures."""
import csv
import json
import math
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F


def ssim(x, y):
    q = torch.arange(-5, 6, device=x.device, dtype=x.dtype)
    w = torch.softmax(-q.square()/(2*1.5**2), 0)
    kernel = (w[:, None]*w[None, :])[None, None]
    def avg(a):
        return F.conv2d(a, kernel)
    a, b = avg(x), avg(y)
    va, vb = avg(x*x)-a*a, avg(y*y)-b*b
    cov = avg(x*y)-a*b
    return (((2*a*b+.01**2)*(2*cov+.03**2)) /
            ((a*a+b*b+.01**2)*(va+vb+.03**2))).mean((1,2,3))


def grid(path, rows, labels, titles=None, caption=""):
    n, m = len(rows), len(rows[0])
    fig, axes = plt.subplots(n, m, figsize=(1.25*m+1.4, 1.25*n+.5), squeeze=False)
    if caption:
        fig.suptitle(caption)
    for i, (samples, label) in enumerate(zip(rows, labels)):
        for j, ax in enumerate(axes[i]):
            ax.imshow(samples[j].cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center")
            if i == 0 and titles:
                ax.set_title(titles[j], fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def generate(models, name, n, device):
    m = models[name]
    if name == "DDPM":
        return m.sample(n)
    if name == "VAE":
        return m.decode(torch.randn(n, 64, device=device))
    return m.g(torch.randn(n, m.z_dim, device=device))


def fid(models, dl, device, count):
    names = ["VAE","GAN","DCGAN","DDPM"]
    empty = dict.fromkeys(names)
    if count == 0:
        return empty, "Disabled for this run"
    if not 2 <= count <= len(dl.dataset):
        raise ValueError("FID count must be between 2 and the size of the test split")
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        metric = FrechetInceptionDistance(feature=2048, normalize=False,
                                           reset_real_features=False).to(device)
    except Exception as error:
        return empty, f"FID extractor unavailable: {type(error).__name__}: {error}"
    def rgb(x):
        return (x*255).round().to(torch.uint8).repeat(1,3,1,1)
    seen = 0
    for x, _ in dl:
        x = x[:count-seen].to(device)
        metric.update(rgb(x), real=True)
        seen += len(x)
        if seen == count:
            break
    result = {}
    for name in names:
        metric.reset()
        for start in range(0, count, 64):
            metric.update(rgb(generate(models,name,min(64,count-start),device)), real=False)
        result[name] = metric.compute().item()
    return result, "torchmetrics FID / torch-fidelity Inception pool3, identical real test images"


@torch.inference_mode()
def evaluate(models, dl, device, out, history, fid_count):
    from run import sync, dump
    torch.manual_seed(42)
    mode = json.loads((out/"manifest.json").read_text())["mode"]
    caption = {"quick":"Fashion-MNIST subset · diagnostic", "smoke":"Synthetic noise · smoke check",
               "full":"Fashion-MNIST · full split"}[mode]
    clf = models["MLP"]
    names = ["AE","SparseAE","VAE","TransposeClass","TransposeLatent","LatentDecoder"]
    metrics = {k:{} for k in list(models)+names}
    sums = {k: [0., 0.] for k in names}
    def reconstruct(x, y):
        z = clf.encode(x)
        return {**{k:models[k](x)[0] for k in names[:3]}, "TransposeClass":clf.transpose_class(y),
                "TransposeLatent":clf.transpose_latent(z), "LatentDecoder":models["LatentDecoder"](z)}
    total = correct = 0
    inactive = {"AE":0, "SparseAE":0}
    for x, y in dl["test"]:
        x,y = x.to(device),y.to(device)
        pred = reconstruct(x,y)
        for name in inactive:
            z = models[name].encode(x)[0]
            inactive[name] += (z.abs()<.001).sum().item()
        for k, v in pred.items():
            sums[k][0] += (v-x).square().mean((1,2,3)).sum().item()
            sums[k][1] += ssim(x,v).sum().item()
        correct += (clf(x).argmax(1)==y).sum().item()
        total += len(x)
    for k, (err, sim) in sums.items():
        mse = err/total
        metrics[k].update(mse=mse, psnr_db=-10*math.log10(max(mse,1e-30)), ssim=sim/total)
    metrics["MLP"]["test_accuracy"] = correct/total
    for name, n in inactive.items():
        metrics[name]["test_latent_fraction_abs_below_0_001"] = n/(total*64)
    diagnostics = {}
    for key, layer in (("W1",clf.a), ("W2",clf.b), ("W3",clf.c)):
        w = layer.weight.detach().cpu().double()
        gram = w.T @ w
        diagnostics[key] = dict(shape=list(w.shape),rank=int(torch.linalg.matrix_rank(w)),
                                input_gram_frobenius_error=float((gram-torch.eye(w.size(1))).norm()))
    first_batch, first_labels = next(iter(dl["test"]))
    pair_ix = next(i for i in range(1,len(first_labels)) if first_labels[i]!=first_labels[0])
    endpoint_ids = [int(dl["test"].dataset.indices[k]) for k in (0,pair_ix)]
    endpoint_labels = first_labels[[0,pair_ix]].tolist()
    pair = first_batch[[0,pair_ix]].to(device)
    x,y = first_batch[:8].to(device),first_labels[:8].to(device)
    pred = reconstruct(x,y)
    grid(out/"01_reconstructions.png",[x]+[pred[k] for k in names[:3]], ["Original"]+names[:3],caption=caption)
    random_rows = []
    for split in ["train","test"]:
        latents = torch.cat([clf.encode(a.to(device)).cpu() for a,_ in dl[split]],0)
        mu, sd = latents.mean(0), latents.std(0,unbiased=False)
        torch.save(dict(mean=mu,std=sd),out/("latent_moments_"+split+".pt"))
        z = (mu + torch.randn(8,64)*sd).to(device)
        random_rows.append(models["LatentDecoder"](z))
    grid(out/"03_mlp.png", [x,pred["TransposeClass"],pred["TransposeLatent"],
         pred["LatentDecoder"],pred["AE"]]+random_rows, ["Original","Class transpose",
         "Latent transpose","Learned decoder","AE","Train prior","Test prior (oracle)"],caption=caption)
    fig, axes = plt.subplots(2,2,figsize=(7,7))
    fig.suptitle(caption)
    for ax,name in zip(axes.flat,["VAE","GAN","DCGAN","DDPM"]):
        samples=generate(models,name,16,device).reshape(4,4,28,28).permute(0,2,1,3).reshape(112,112)
        ax.imshow(samples.cpu(),cmap="gray",vmin=0,vmax=1)
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out/"02_prior_samples.png",dpi=140)
    plt.close(fig)
    alpha = torch.linspace(0,1,6,device=device)[:,None]
    rows = []
    for name in ("AE","SparseAE","VAE","GAN","DCGAN"):
        m = models[name]
        adv = name in ("GAN","DCGAN")
        endpoints = torch.randn(2,m.z_dim,device=device) if adv else m.encode(pair)[0]
        rows.append((m.g if adv else m.decode)(torch.lerp(endpoints[0],endpoints[1],alpha)))
    grid(out/"04_interpolation.png",rows,["AE","SparseAE","VAE","GAN (noise)","DCGAN (noise)"],
         [f"α={i/5:.1f}" for i in range(6)],caption)
    trace = []
    models["DDPM"].sample(1, trace)
    fig, axes = plt.subplots(3,3,figsize=(12,9))
    fig.suptitle(caption)
    for ax,(name,model) in zip(axes.flat,models.items()):
        h = [row for row in history if row["model"]==name]
        for key in ("train","validation"):
            ax.plot([r["epoch"] for r in h],[r[key]["loss"] for r in h],marker=".",label=key)
        if "discriminator" in h[0]["train"]:
            ax.plot([r["epoch"] for r in h],[r["train"]["discriminator"] for r in h],label="train D")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
        times = [r["seconds"] for r in h]
        vals = [r["train"]["loss"] for r in h]
        metrics[name].update(mean_training_seconds_per_epoch=statistics.mean(times),
                             train_loss_sd=statistics.pstdev(vals) if len(vals)>1 else None)
    ax = axes.flat[-1]
    ax.imshow(torch.cat([v[0,0] for k,v in trace],1),cmap="gray",vmin=0,vmax=1)
    ax.axis("off")
    ax.set_title("Reverse t = "+", ".join(str(k) for k,v in trace),fontsize=8)
    fig.tight_layout()
    fig.savefig(out/"05_training_and_reverse.png",dpi=140)
    plt.close(fig)
    timing_x = x.repeat(8,1,1,1)
    for name in models:
        def infer():
            if name in ("GAN","DCGAN","DDPM","VAE"):
                return generate(models,name,64,device)
            if name=="LatentDecoder":
                return models[name](clf.encode(timing_x))
            return models[name](timing_x)
        infer()
        durations=[]
        for _ in range(3):
            sync(device)
            t=time.perf_counter()
            infer()
            sync(device)
            durations.append((time.perf_counter()-t)*1000)
        metrics[name]["median_inference_ms_batch64"] = statistics.median(durations)
    scores, status = fid(models,dl["test"],device,fid_count)
    for name,value in scores.items():
        metrics[name]["fid"] = value
    dump(out/"metrics.json",dict(models=metrics,fid_requested_count=fid_count,fid_status=status,
                                 interpolation_image_ids=endpoint_ids,interpolation_labels=endpoint_labels,
                                 transpose_diagnostics=diagnostics,evaluation_seed=42))
    keys = sorted(set().union(*(v.keys() for v in metrics.values())))
    with (out/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,["model"]+keys)
        writer.writeheader()
        writer.writerows(dict(model=k,**v) for k,v in metrics.items())
