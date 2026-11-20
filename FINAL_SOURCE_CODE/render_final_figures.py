"""Regenerate the five report figures directly from held-out images and saved weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageStat
from torch.utils.data import DataLoader, Subset

from data import FILES, Images, read_idx
from evaluate import ssim
from models import AE, Adversary, Classifier, DDPM, decoder

LABELS = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
NAMES = ["AE", "SparseAE", "VAE", "GAN", "DCGAN", "DDPM"]
HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def verify_images(args, names):
    for name in names:
        path = args.out/name
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            if min(img.size)<300 or ImageStat.Stat(img.convert("L")).stddev[0]<5:
                raise ValueError(f"Empty or undersized figure: {path}")


def checkpoint(model, path, provenance):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state"])
    provenance[path.stem] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "epoch": ckpt["epoch"]}
    return model.eval().requires_grad_(False)


def show_rows(path, rows, labels, headings=None):
    fig, axes = plt.subplots(len(rows), len(rows[0]), figsize=(10, len(rows)*.94+.35),
                             squeeze=False, layout="constrained")
    for i, (row, label) in enumerate(zip(rows, labels)):
        axes[i,0].set_ylabel(label, ha="right", va="center", rotation=0, fontsize=10)
        for j, image in enumerate(row):
            ax = axes[i,j]
            ax.imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0 and headings:
                ax.set_title(headings[j], fontsize=9)
    fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def read_data(data):
    data.mkdir(parents=True, exist_ok=True)
    raw = data/"FashionMNIST"/"raw"
    if not all((data/name).is_file() for name in FILES) and all((raw/name).is_file() for name in FILES):
        data = raw
    parts = [read_idx(data, k) for k in FILES]
    return Images(torch.cat(parts[::2]), torch.cat(parts[1::2]))


def proof(args, dataset, indices, clf):
    compare = torch.load(args.proof/"indices.pt", weights_only=True)
    classifier_indices = torch.load(args.classifier/"indices.pt", weights_only=True)
    for split in indices:
        assert torch.equal(compare[split], classifier_indices[split])
    merged = torch.cat(list(compare.values())).sort().values
    assert torch.equal(merged, torch.arange(len(dataset)))
    test = DataLoader(Subset(dataset, compare["test"].tolist()), batch_size=128)
    prov = {}
    ae = checkpoint(AE(), args.proof/"AE.pt", prov)
    dec = checkpoint(decoder(), args.proof/"LatentDecoder.pt", prov)
    labels = ["Transpos kelas","Transpos laten","Decoder terlatih","AE 64D"]

    def predictions(x, y):
        h = clf.encode(x)
        return [clf.transpose_class(y), clf.transpose_latent(h), dec(h), ae(x)[0]]

    sums = torch.zeros(4,2,dtype=torch.double)
    count = correct = 0
    for x, y in test:
        pred = predictions(x, y)
        correct += int((clf(x).argmax(1)==y).sum())
        for index,v in enumerate(pred):
            sums[index,0] += (v-x).square().mean((1,2,3)).double().sum()
            sums[index,1] += ssim(x,v).double().sum()
        count += len(x)
    metrics={}
    for name,(sq,sim) in zip(["TransposeClass","TransposeLatent","LatentDecoder","AE"], sums.tolist()):
        mse = sq/count
        metrics[name] = dict(mse=mse, ssim=sim/count, psnr_db=-10*math.log10(mse))
    cached = read(args.proof/"metrics.json")
    for name, result in metrics.items():
        for key, value in result.items():
            assert abs(value-cached["models"][name][key]) < 1e-5
    assert abs(correct/count-cached["classifier_test_accuracy"]) < 1e-6 and correct/count >= .88
    x,y = next(iter(test))
    x,y = x[:8],y[:8]
    rows = [x] + predictions(x,y)
    g = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(8,64,generator=g)
    moment_check = {}
    prior_diagnostics = {}
    audit_noise = torch.randn(1024,64,generator=g)
    for split in ["train","test"]:
        mom = torch.load(args.proof/("latent_moments_"+split+".pt"), weights_only=True)
        subset = DataLoader(Subset(dataset, compare[split].tolist()), batch_size=256)
        latent = torch.cat([clf.encode(batch) for batch,_ in subset])
        errors = dict(mean=float((mom["mean"]-latent.mean(0)).abs().max()),
                      std=float((mom["std"]-latent.std(0,unbiased=False)).abs().max()))
        assert max(errors.values()) < 2e-5
        moment_check[split] = errors
        prior = mom["mean"]+audit_noise*mom["std"]
        prior_diagnostics[split] = dict(n=len(prior),fraction_negative=float((prior<0).float().mean()),
            examples_with_negative=int((prior<0).any(1).sum()),empirical_negatives=int((latent<0).sum()))
        rows.append(dec(mom["mean"]+noise*mom["std"]))
    show_rows(args.out/"03_mlp.png", rows, ["Citra uji"]+labels+["Prior latih","Prior uji (oracle)"],
              [LABELS[i] for i in y.tolist()])
    write(args.log/"proof.json", dict(models=metrics, classifier_correct=correct,n_test=count,
                                      classifier_accuracy=correct/count,
                                      classifier_sha256=hashlib.sha256((args.classifier/"MLP.pt").read_bytes()).hexdigest(),
                                      checkpoints=prov,figure_indices=compare["test"][:8].tolist(),moment_verification_max_abs=moment_check,
                                      prior_support_check=prior_diagnostics,
                                      prior_rows="Same seed and noise; unpaired alternate samples using diagonal Gaussian moments. Test moments use held-out information."))


def tile(x):
    return x[:,0].reshape(4,4,28,28).permute(0,2,1,3).reshape(112,112)


def adjacent_contrast(x):
    """Mean absolute horizontal and vertical intensity difference, per image."""
    dx = (x[:,:,:,1:]-x[:,:,:,:-1]).abs().mean((1,2,3))
    dy = (x[:,:,1:,:]-x[:,:,:-1,:]).abs().mean((1,2,3))
    return (dx+dy)/2


def render_history(args, history, trace):
    fig, axes = plt.subplots(2,3,figsize=(11.8,5.8), layout="constrained")
    def curve(ax, name, field, label, color, style="-", split="train", fallback=None):
        hh = [h for h in history if h["model"]==name]
        vals = [h[split].get(field,h[split].get(fallback)) for h in hh]
        assert hh and all(v is not None for v in vals), (name, field, split)
        ax.plot([h["epoch"] for h in hh], vals, style, color=color, marker=".", markersize=3,label=label)
    a,b,c,d,e,f = axes.flat
    colors = ["#23699c","#b34e23","#227859"]
    for name,col in zip(NAMES[:3],colors):
        curve(a,name,"mse",name,col,fallback="loss")
    a.set_title("Rekonstruksi latih · MSE")
    h_vae = [h for h in history if h["model"]=="VAE"]
    epochs = [h["epoch"] for h in h_vae]
    total = torch.tensor([h["train"]["loss"] for h in h_vae])
    kl = torch.tensor([h["train"]["kl"]/784 for h in h_vae])
    for vals,label,col in [(total,"Total",colors[0]),(total-kl,"Rekonstruksi BCE",colors[1]),(kl,"KL",colors[2])]:
        b.plot(epochs,vals,marker=".",markersize=3,label=label,color=col)
    b.set_title("VAE · komponen latih per piksel")
    for ax,name in [(c,"GAN"),(d,"DCGAN")]:
        curve(ax,name,"loss","Generator",colors[0])
        curve(ax,name,"discriminator","Diskriminator",colors[1])
        ax.set_title(name+" · objektif latih")
    curve(e,"DDPM","loss","Latih",colors[0])
    curve(e,"DDPM","loss","Validasi",colors[1],"--",split="validation")
    e.set_title("DDPM · MSE noise")
    for ax in [a,b,c,d,e]:
        ax.legend(frameon=False,fontsize=8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.18)
        ax.xaxis.get_major_locator().set_params(integer=True)
    chosen = [(k,v) for k,v in trace if k!=999]
    f.imshow(torch.cat([v[0,0] for _,v in chosen],1),cmap="gray",vmin=0,vmax=1,interpolation="nearest")
    f.set_yticks([])
    f.set_xticks([13.5+28*i for i in range(len(chosen))],[str(k) for k,_ in chosen],fontsize=8)
    f.set_xlabel("Langkah tersisa")
    f.set_title("DDPM · satu lintasan balik")
    fig.savefig(args.out/"05_training_and_reverse.png",dpi=170,bbox_inches="tight")
    plt.close(fig)


def sampling(args, dataset, indices, clf):
    prov = {}
    model_list = [AE(),AE(),AE(True)]
    if args.stage!="reconstruction":
        model_list.extend([Adversary(False),Adversary(True),DDPM(1000)])
    models = {name:checkpoint(net,args.run/(name+".pt"),prov) for name,net in zip(NAMES,model_list)}
    test_indices = indices["test"][:16].tolist()
    x = torch.stack([dataset[i][0] for i in test_indices])
    y = [int(dataset[i][1]) for i in test_indices]
    contrast_sum = dict.fromkeys(["Citra uji"] + NAMES[:3],0.)
    batches = DataLoader(Subset(dataset,indices["test"].tolist()),batch_size=128)
    for batch,_ in batches:
        for name in contrast_sum:
            result = batch if name=="Citra uji" else models[name](batch)[0]
            contrast_sum[name] += adjacent_contrast(result).double().sum().item()
    contrast = {name:value/len(batches.dataset) for name,value in contrast_sum.items()}
    show_rows(args.out/"01_reconstructions.png", [x[:8]]+[models[n](x[:8])[0] for n in NAMES[:3]],
              ["Citra uji"]+NAMES[:3], [LABELS[i] for i in y[:8]])
    if args.stage=="reconstruction":
        write(args.log/"adjacent_contrast.json",dict(checkpoints=prov,heldout_mean_adjacent_contrast=contrast,n=len(batches.dataset)))
        return
    torch.manual_seed(args.seed)
    samples = {"Citra uji":x, "VAE":models["VAE"].decode(torch.randn(16,64))}
    for n in NAMES[3:5]:
        samples[n] = models[n].g(torch.randn(16,models[n].z_dim))
    trace=[]
    samples["DDPM"]=models["DDPM"].sample(16,trace)
    fig, axes = plt.subplots(1,5,figsize=(12,2.6),layout="constrained")
    for ax,(name,batch) in zip(axes,samples.items()):
        ax.imshow(tile(batch),cmap="gray",vmin=0,vmax=1,interpolation="nearest")
        ax.axis("off")
        ax.set_title(name,fontsize=13)
    fig.savefig(args.out/"02_prior_samples.png",dpi=170,bbox_inches="tight",pad_inches=.04)
    plt.close(fig)
    torch.manual_seed(args.seed)
    # Across a garment and footwear the contrast is easy to inspect on 28×28 images.
    garment = next(j for j,label in enumerate(y) if label in (0,1,2,3,4,6))
    footwear = next(j for j,label in enumerate(y) if label in (5,7,9))
    pair = [garment,footwear]
    alpha=torch.linspace(0,1,6)[:,None]
    rows=[]
    notes={}
    for n in NAMES[:5]:
        m=models[n]
        if n in NAMES[:3]:
            z=m.encode(x[pair])[0]
            rows.append(m.decode(torch.lerp(z[0],z[1],alpha)))
        else:
            pool=torch.randn(256,m.z_dim)
            imgs=m.g(pool)
            confid,preds=clf(imgs).softmax(1).max(1)
            first=int(confid.argmax())
            different=(preds!=preds[first])
            if different.any():
                last=int(confid.masked_fill(~different,-1).argmax())
            else:
                # A model may yield one predicted class for every candidate.
                last=int((pool-pool[first]).square().sum(1).argmax())
            z=pool[[first,last]]
            images=m.g(torch.lerp(z[0],z[1],alpha))
            rows.append(images)
            notes[n]=dict(pool_size=len(pool),n_predicted_classes=len(set(preds.tolist())),
                endpoint_ids=[first,last],predicted_labels=preds[[first,last]].tolist(),
                predicted_names=[LABELS[i] for i in preds[[first,last]].tolist()],
                confidences=confid[[first,last]].tolist(),distinct=bool(different.any()),
                selection="Maximum softmax, then maximum softmax from a different predicted class when available")
    row_labels = NAMES[:3]+[name+"\nPrediksi MLP\n"+" → ".join(notes[name]["predicted_names"]) for name in NAMES[3:5]]
    show_rows(args.out/"04_interpolation.png",rows,row_labels,[f"α = {a:.1f}" for a in alpha.flatten()])
    render_history(args,read(args.run/"history.json"),trace)
    write(args.log/"figures.json",dict(seed=args.seed,checkpoints=prov,real_ids=test_indices,real_labels=y,
        interpolation_source_ids=[test_indices[i] for i in pair],interpolation_source_labels=[y[i] for i in pair],
        generated_endpoints=notes,panel_predictions={k:clf(v).argmax(1).tolist() for k,v in samples.items()},
        heldout_mean_adjacent_contrast=contrast,n_heldout=len(batches.dataset),
        interpolation_adjacent_rmse={k:(v[1:]-v[:-1]).square().mean((1,2,3)).sqrt().tolist() for k,v in zip(NAMES,rows)}))


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,default=HERE/"runs/full_budget")
    p.add_argument("--proof",type=Path,default=HERE/"runs/classifier_proof_full")
    p.add_argument("--classifier",type=Path,default=HERE/"runs/classifier_full")
    p.add_argument("--data",type=Path,default=HERE/"data")
    p.add_argument("--out",type=Path,default=HERE.parent/"FINAL_FIGURES")
    p.add_argument("--log",type=Path,default=HERE/"runs/figure_verification")
    p.add_argument("--stage",choices=["all","proof","samples","reconstruction"],default="all")
    p.add_argument("--threads",type=int,default=2)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    args.log.mkdir(exist_ok=True,parents=True)
    args.out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(args.threads)
    clf=checkpoint(Classifier(),args.classifier/"MLP.pt",{})
    indices=torch.load((args.proof if args.stage=="proof" else args.run)/"indices.pt",weights_only=True)
    dataset=read_data(args.data)
    generation_split = torch.randperm(len(dataset),generator=torch.Generator().manual_seed(42))
    train_end, val_end = 8*len(dataset)//10, 9*len(dataset)//10
    expected = dict(train=generation_split[:train_end],validation=generation_split[train_end:val_end],
                    test=generation_split[val_end:])
    for split, ids in indices.items():
        if not torch.equal(ids, expected[split][:len(ids)]):
            raise ValueError(f"Unrecognized seed-42 {split} indices")
    if args.stage in ["all","proof"]:
        proof(args,dataset,indices,clf)
        verify_images(args,["03_mlp.png"])
    if args.stage in ["all","samples","reconstruction"]:
        sampling(args,dataset,indices,clf)
        names=["01_reconstructions.png"]
        if args.stage!="reconstruction":
            names.extend(["02_prior_samples.png","04_interpolation.png","05_training_and_reverse.png"])
        verify_images(args,names)

if __name__=="__main__":
    main()
