"""Full-split, validation-selected training and independent evaluation for six families."""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path

import torch

from data import loaders
from models import AE, Adversary, Classifier, DDPM
from run import epoch, sync, weight_hash


NAMES = ("AE","SparseAE","VAE","GAN","DCGAN","DDPM")
FACTORIES = dict(AE=AE,SparseAE=AE,VAE=lambda:AE(True),GAN=lambda:Adversary(False),
                 DCGAN=lambda:Adversary(True),DDPM=DDPM)


def write_json(path, payload):
    tmp = path.with_suffix(".new")
    tmp.write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    os.replace(tmp,path)


def read_json(path, fallback):
    return json.loads(path.read_text()) if path.exists() else fallback


def split_fingerprint(dls):
    h=hashlib.sha256()
    for name in ("train","validation","test"):
        h.update(torch.tensor(dls[name].dataset.indices,dtype=torch.int64).numpy().tobytes())
    return h.hexdigest()


class Batches:
    """Cache fixed data once; deterministic training permutations are independent of model RNG."""
    def __init__(self,dl,device,shuffle=False):
        images,labels=dl.dataset.dataset.tensors
        self.x=(images[dl.dataset.indices].to(device).float()/255).contiguous()
        self.y=labels[dl.dataset.indices].to(device).long()
        self.shuffle=shuffle
        self.generator=torch.Generator().manual_seed(42)

    def __iter__(self):
        if self.shuffle:
            idx=torch.randperm(len(self.x),generator=self.generator).to(self.x.device)
            for item in idx.split(128):
                yield self.x[item],self.y[item]
        else:
            yield from zip(self.x.split(128),self.y.split(128))


@torch.no_grad()
def feature_moments(clf, xs):
    z=torch.cat([clf.encode(x) for x in xs])
    return z.mean(0),z.std(0,unbiased=False)


def save_state(path,model,e,score):
    tmp=path.with_suffix(".tmp")
    torch.save(dict(state=model.state_dict(),epoch=e,validation_score=score),tmp)
    os.replace(tmp,path)


def train(args,device,dls,out,manifest):
    batches={k:Batches(dls[k],device,k=="train") for k in ("train","validation")}
    clf=Classifier().to(device)
    clf.load_state_dict(torch.load(args.classifier,weights_only=True,map_location=device)["state"])
    clf.eval().requires_grad_(False)
    split_path=args.classifier.parent/"indices.pt"
    if split_path.exists():
        reference=torch.load(split_path,map_location="cpu",weights_only=True)
        if any(reference[k].tolist()!=dls[k].dataset.indices for k in dls):
            raise ValueError("Frozen feature classifier was trained with a different split")
    manifest["validation_feature_classifier_sha256"]=weight_hash(clf)
    real_mean,real_std=feature_moments(clf,batches["validation"].x.split(128))
    run_stats=read_json(out/"training.json",{})
    hist=read_json(out/"history.json",[])
    for name in args.models:
        if name in run_stats and (out/(name+".pt")).exists():
            continue
        hist=[r for r in hist if r["model"]!=name]
        torch.manual_seed(42)
        batches["train"].generator.manual_seed(42)
        m=FACTORIES[name]().to(device)
        adv=name in ("GAN","DCGAN")
        opts=[torch.optim.Adam(net.parameters(),lr=.0002,betas=(.5,.999)) for net in (m.g,m.d)] if adv else [
            torch.optim.Adam(m.parameters(),lr=.001)]
        val_noise=None
        if adv:
            val_noise=torch.randn(len(batches["validation"].x),m.z_dim,
                                  generator=torch.Generator().manual_seed(42)).to(device)
        best=math.inf
        reference=math.inf
        patience=0
        times=[]
        budget=args.seconds_per_model
        started=time.perf_counter()
        max_epochs=args.max_epochs
        stop="epoch_cap"
        for e in range(1,max_epochs+1):
            sync(device)
            start=time.perf_counter()
            tr=epoch(name,m,batches["train"],None,device,opts)
            sync(device)
            sec=time.perf_counter()-start
            times.append(sec)
            with torch.random.fork_rng(devices=[device.index or 0] if device.type=="cuda" else []):
                torch.manual_seed(42)
                val=epoch(name,m,batches["validation"],None,device)
                if adv:
                    with torch.inference_mode():
                        mu,sd=feature_moments(clf,(m.g(z) for z in val_noise.split(128)))
                        score=((mu-real_mean).square()+(sd-real_std).square()).mean().item()
                    val["classifier_moment_distance"]=score
                else:
                    score=val["loss"]
            if score<best:
                best=score
                best_epoch=e
                save_state(out/(name+".pt"),m,e,best)
            if score<reference*(1-args.relative_delta):
                reference=score
                patience=0
            else:
                patience+=1
            hist.append(dict(model=name,epoch=e,train=tr,validation=val,seconds=sec))
            write_json(out/"history.json",hist)
            print(f"{name} epoch={e} val={score:.6f} seconds={sec:.2f} best_epoch={best_epoch}",flush=True)
            elapsed=time.perf_counter()-started
            if e>=args.min_epochs and patience>=args.patience:
                stop="validation_plateau"
                break
            if e>=args.min_epochs and elapsed + elapsed/e >budget:
                stop="measured_wall_time_budget"
                break
        run_stats[name]=dict(epochs_trained=e,selected_epoch=best_epoch,stop_reason=stop,
            validation_criterion="frozen_classifier_diagonal_moment_distance" if adv else "validation_loss",
            best_validation_score=best,parameters=sum(p.numel() for p in m.parameters()),
            training_device=str(device),training_hardware=torch.cuda.get_device_name(device) if device.type=="cuda" else platform.processor(),
            torch_version=str(torch.__version__),training_threads=args.threads,
            min_epochs=args.min_epochs,max_epochs=args.max_epochs,
            seconds_per_model=budget,patience=args.patience,relative_delta=args.relative_delta,
            latent_dim=64 if name in ("AE","SparseAE","VAE","GAN") else 100 if name=="DCGAN" else None,
            mean_training_seconds_per_epoch=statistics.mean(times),total_training_seconds=sum(times),
            training_wall_seconds=time.perf_counter()-started)
        write_json(out/"training.json",run_stats)
    manifest["completed_models"]=list(run_stats)
    manifest["status"]="training_complete" if set(run_stats)==set(NAMES) else "partial_training"
    write_json(out/"manifest.json",manifest)


@torch.inference_mode()
def evaluate(args,device,dls,out):
    import hashlib

    from evaluate import ssim, generate
    stats=read_json(out/"training.json",{})
    results=read_json(out/"metrics.json",dict(models={},fid_status="pending",fid_requested_count=0))
    test=Batches(dls["test"],device)
    results.update(test_count=len(test.x),benchmark_batch_size=64,
                   checkpoint_sha256=results.get("checkpoint_sha256", {}))
    models={}
    for name in args.models:
        if name not in stats:
            raise ValueError(f"No finished training for {name}")
        m=FACTORIES[name]().to(device)
        checkpoint_path=out/(name+".pt")
        checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=True)
        if checkpoint["epoch"]!=stats[name]["selected_epoch"]:
            raise ValueError(f"Checkpoint and recorded validation selection disagree for {name}")
        m.load_state_dict(checkpoint["state"])
        results["checkpoint_sha256"][name]=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        m.eval().requires_grad_(False)
        models[name]=m
        row=dict(stats[name])
        if name in ("AE","SparseAE","VAE"):
            err=sim=inactive=latent_abs=0.
            for x,_ in test:
                pred,z,_=m(x)
                err+=(x-pred).square().mean((1,2,3)).sum().item()
                sim+=ssim(x,pred).sum().item()
                inactive+=(z.abs()<.001).sum().item()
                latent_abs+=z.abs().sum().item()
            n=len(test.x)
            row.update(mse=err/n,psnr_db=-10*math.log10(err/n),ssim=sim/n)
            if name!="VAE":
                row["test_latent_fraction_abs_below_0_001"]=inactive/(64*n)
                row["test_latent_mean_abs"]=latent_abs/(64*n)
        torch.manual_seed(42)
        def infer(size):
            return m(test.x[:size]) if name in ("AE","SparseAE") else generate(models,name,size,device)
        infer(64)
        durations=[]
        for repeat in range(3):
            sync(device)
            t=time.perf_counter()
            infer(64)
            sync(device)
            durations.append(1000*(time.perf_counter()-t))
        row.update(median_inference_ms_batch64=statistics.median(durations),latency_repeats_ms=durations,
                   evaluation_device=str(device),evaluation_hardware=torch.cuda.get_device_name(device) if device.type=="cuda" else platform.processor())
        if name in ("VAE","GAN","DCGAN"):
            for _ in range(5):
                infer(1)
            single=[]
            for _ in range(30):
                sync(device)
                started=time.perf_counter()
                infer(1)
                sync(device)
                single.append(1000*(time.perf_counter()-started))
            row["local_batch1_p95_ms"]=statistics.quantiles(single,n=20,method="inclusive")[18]
            row["local_batch1_repeats"]=len(single)
        results["models"][name]=row
        write_json(out/"metrics.json",results)
        print(f"{name} benchmark latency64 = {statistics.median(durations):.2f} ms",flush=True)
    if args.fid_count:
        from fid_benchmark import score_models
        classifier=None
        if args.classifier.is_file():
            classifier=Classifier().to(device)
            proof=torch.load(args.classifier,map_location=device,weights_only=True)
            classifier.load_state_dict(proof["state"])
            classifier.eval().requires_grad_(False)
            expected_hash=read_json(out/"manifest.json",{}).get("validation_feature_classifier_sha256")
            if expected_hash and weight_hash(classifier)!=expected_hash:
                raise ValueError("The class-coverage probe differs from the validation feature classifier")
        score_models(models,test.x,args.fid_count,device,out,results,classifier)
    write_json(out/"metrics.json",results)
    keys=sorted(set().union(*(r.keys() for r in results["models"].values()))
                -{"latency_repeats_ms","predicted_class_histogram"})
    with (out/"metrics.csv").open("w",newline="",encoding="utf-8") as handle:
        w=csv.DictWriter(handle,["model"]+keys,extrasaction="ignore")
        w.writeheader()
        w.writerows(dict(model=k,**v) for k,v in results["models"].items())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("runs/full_budget"))
    p.add_argument("--data",default="data")
    p.add_argument("--classifier",type=Path,default=Path("../FINAL_CHECKPOINTS/classifier_full/MLP.pt"))
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--models",nargs="+",choices=NAMES,default=list(NAMES))
    p.add_argument("--phase",choices=("train","evaluate","all"),default="all")
    p.add_argument("--max-epochs",type=int,default=30)
    p.add_argument("--min-epochs",type=int,default=3)
    p.add_argument("--seconds-per-model",type=float,default=240)
    p.add_argument("--patience",type=int,default=4)
    p.add_argument("--relative-delta",type=float,default=.002)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--fid-count",type=int,default=0)
    a=p.parse_args()
    if not (1<=a.min_epochs<=a.max_epochs and a.seconds_per_model>0 and a.patience>=1 and 0<=a.relative_delta<1):
        p.error("Invalid training budget or validation stopping configuration")
    if a.fid_count and not 2<=a.fid_count<=7000:
        p.error("Choose between 2 and 7000 FID samples")
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device(a.device)
    dls=loaders(a.data,a.out,"full")
    manifest=read_json(a.out/"manifest.json",{})
    split_hash=split_fingerprint(dls)
    if manifest.get("split_sha256",split_hash)!=split_hash:
        raise ValueError("Full split hash differs from previous training")
    if not manifest:
        manifest=dict(mode="full",seed=42,ddpm_steps=1000,batch_size=128,device=str(device),
            platform=platform.platform(),python=platform.python_version(),torch=str(torch.__version__),
            hardware=torch.cuda.get_device_name(device) if device.type=="cuda" else platform.processor(),
            counts={k:len(v.dataset) for k,v in dls.items()},max_epochs=a.max_epochs,min_epochs=a.min_epochs,
            seconds_per_model=a.seconds_per_model,patience=a.patience,relative_delta=a.relative_delta,
            real_test_used_for_selection=False,threads=a.threads,status="training",lr=.001,gan_lr=.0002,
            gan_betas=[.5,.999],sparse_weight=.001,vae_kl_weight=1,ddpm_beta=[.0001,.02])
        write_json(a.out/"manifest.json",manifest)
    manifest["split_sha256"]=split_hash
    if a.phase!="evaluate":
        train(a,device,dls,a.out,manifest)
    if a.phase!="train":
        evaluate(a,device,dls,a.out)


if __name__=="__main__":
    main()
