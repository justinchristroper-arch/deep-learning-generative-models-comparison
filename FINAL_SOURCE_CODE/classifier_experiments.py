"""Complete the inversion experiment using an independently trained, qualified classifier."""
import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path

import torch

from data import loaders
from evaluate import grid, ssim
from models import AE, Classifier, decoder
from run import dump, epoch, sync, weight_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classifier",type=Path,default=Path("runs/classifier_full"))
    p.add_argument("--out",type=Path,default=Path("runs/classifier_proof_full"))
    p.add_argument("--data",default="data")
    p.add_argument("--device",default=None)
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--threads",type=int,default=2)
    args = p.parse_args()
    original = json.loads((args.classifier/"manifest.json").read_text())
    accuracy = json.loads((args.classifier/"classifier_metrics.json").read_text())
    if original["mode"] != "full" or accuracy["test_accuracy"] < .88:
        p.error("Supply a full-split classifier with independently recorded test accuracy at or above 88%")
    if args.epochs < 1:
        p.error("Choose at least one epoch")
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        p.error("Choose an empty output directory to preserve previous results")
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    device=torch.device(args.device or original["device"])
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    dl=loaders(args.data,out,"full")
    old=torch.load(args.classifier/"indices.pt",weights_only=True)
    for k,v in dl.items():
        if old[k].tolist()!=v.dataset.indices:
            raise ValueError(f"Classifier split differs for {k}")
    clf=Classifier().to(device)
    checkpoint=torch.load(args.classifier/"MLP.pt",map_location=device,weights_only=True)
    if checkpoint["epoch"]!=accuracy["selected_epoch"]:
        raise ValueError("Selected classifier epoch does not match accuracy report")
    clf.load_state_dict(checkpoint["state"])
    clf.eval().requires_grad_(False)
    frozen=weight_hash(clf)
    shutil.copy2(args.classifier/"classifier_metrics.json",out/"classifier_metrics_source.json")
    manifest=dict(mode="full",dataset="Fashion-MNIST",device=str(device),seed=42,epochs=args.epochs,
                  batch_size=128,lr=.001,cpu_threads=args.threads,torch=str(torch.__version__),
                  counts={k:len(v.dataset) for k,v in dl.items()},classifier_source=args.classifier.name,
                  classifier_selected_epoch=checkpoint["epoch"],classifier_test_accuracy=accuracy["test_accuracy"],
                  classifier_sha256_before_decoder=frozen,indices_verified=True,status="training")
    dump(out/"manifest.json",manifest)
    nets={"AE":AE().to(device),"LatentDecoder":decoder().to(device)}
    history=[]
    for name,net in nets.items():
        opt=torch.optim.Adam(net.parameters(),lr=.001)
        best=math.inf
        for e in range(1,args.epochs+1):
            sync(device)
            start=time.perf_counter()
            train=epoch(name,net,dl["train"],clf,device,[opt])
            sync(device)
            seconds=time.perf_counter()-start
            val=epoch(name,net,dl["validation"],clf,device)
            history.append(dict(model=name,epoch=e,train=train,validation=val,seconds=seconds))
            dump(out/"history.json",history)
            print(f"{name} {e}/{args.epochs} train={train['loss']:.5f} val={val['loss']:.5f}",flush=True)
            if val["loss"]<best:
                best=val["loss"]
                torch.save(dict(state=net.state_dict(),epoch=e),out/(name+".pt"))
        selected=torch.load(out/(name+".pt"),weights_only=True,map_location=device)
        net.load_state_dict(selected["state"])
        net.eval().requires_grad_(False)
        manifest[name+"_selected_epoch"]=selected["epoch"]
    if weight_hash(clf)!=frozen:
        raise RuntimeError("Classifier changed during decoder training")
    manifest["classifier_sha256_after_decoder"]=weight_hash(clf)
    with torch.inference_mode():
        def recon(x,y):
            return dict(TransposeClass=clf.transpose_class(y),TransposeLatent=clf.transpose_latent(clf.encode(x)),
                        LatentDecoder=nets["LatentDecoder"](clf.encode(x)),AE=nets["AE"](x)[0])
        sums={}
        n=0
        for x,y in dl["test"]:
            x,y=x.to(device),y.to(device)
            for k,v in recon(x,y).items():
                pair=sums.setdefault(k,[0.,0.])
                pair[0]+=(x-v).square().mean((1,2,3)).sum().item()
                pair[1]+=ssim(x,v).sum().item()
            n+=len(x)
        result={}
        for k,(sq,sim) in sums.items():
            mse=sq/n
            result[k]=dict(mse=mse,psnr_db=-10*math.log10(max(mse,1e-30)),ssim=sim/n)
        x,y=next(iter(dl["test"]))
        x,y=x[:8].to(device),y[:8].to(device)
        pred=recon(x,y)
        torch.manual_seed(42)
        rows=[x]+list(pred.values())
        for k in ("train","test"):
            z=torch.cat([clf.encode(b.to(device)).cpu() for b,_ in dl[k]])
            mu,sigma=z.mean(0),z.std(0,unbiased=False)
            torch.save(dict(mean=mu,std=sigma),out/("latent_moments_"+k+".pt"))
            noise=torch.randn(8,64)
            rows.append(nets["LatentDecoder"]((mu+sigma*noise).to(device)))
        grid(out/"mlp_full.png",rows,["Original","Class transpose","Latent transpose",
             "Learned decoder","AE","Train prior","Test prior (oracle)"],
             caption="Fashion-MNIST · full split · frozen MLP")
        grams={}
        for key,layer in (("W1",clf.a),("W2",clf.b),("W3",clf.c)):
            w=layer.weight.detach().cpu().double()
            grams[key]=dict(shape=list(w.shape),rank=int(torch.linalg.matrix_rank(w)),
                           input_gram_frobenius_error=float((w.T@w-torch.eye(w.size(1))).norm()))
    dump(out/"metrics.json",dict(models=result,classifier_test_accuracy=accuracy["test_accuracy"],
         transpose_diagnostics=grams,figure_test_indices=dl["test"].dataset.indices[:8],
         figure_test_labels=y.cpu().tolist(),n_test=n))
    with (out/"metrics.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,["model","mse","psnr_db","ssim"])
        writer.writeheader()
        writer.writerows(dict(model=name,**values) for name,values in result.items())
    manifest["status"]="complete"
    dump(out/"manifest.json",manifest)


if __name__=="__main__":
    main()
