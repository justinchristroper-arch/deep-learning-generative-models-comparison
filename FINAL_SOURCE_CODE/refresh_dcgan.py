"""Refresh a completed run's DCGAN and regenerate evaluation from saved checkpoints."""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import torch

from data import loaders
from models import AE, Adversary, Classifier, DDPM, decoder
from run import dump, epoch, sync, weight_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--data", default="data")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--evaluate", action="store_true", help="Evaluate all saved models after DCGAN refresh")
    p.add_argument("--fid-count", type=int, default=None)
    args = p.parse_args()
    out = args.out
    manifest = json.loads((out/"manifest.json").read_text())
    history = json.loads((out/"history.json").read_text())
    revision = out/"dcgan_revision.json"
    device = torch.device(args.device or manifest["device"])
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with tempfile.TemporaryDirectory() as temp:
        dl = loaders(args.data, Path(temp), manifest["mode"])
        saved = torch.load(out/"indices.pt",weights_only=True)
        for key, data in dl.items():
            if data.dataset.indices != saved[key].tolist():
                raise ValueError(f"{key} indices differ from saved experiment")
    if args.evaluate:
        from evaluate import evaluate
        log = json.loads(revision.read_text())
        if log["status"] not in {"checkpoint_ready","evaluated"}:
            raise RuntimeError("Finish DCGAN retraining before evaluation")
        models = dict(MLP=Classifier(), AE=AE(), SparseAE=AE(), VAE=AE(True),
                      GAN=Adversary(False), DCGAN=Adversary(True), DDPM=DDPM(manifest["ddpm_steps"]),
                      LatentDecoder=decoder())
        for name, model in models.items():
            model.load_state_dict(torch.load(out/(name+".pt"),weights_only=True,map_location="cpu")["state"])
            model.to(device).eval().requires_grad_(False)
        if weight_hash(models["DCGAN"]) != log["new_sha256"]:
            raise ValueError("DCGAN checkpoint differs from recorded revision")
        count = args.fid_count
        if count is None:
            count = manifest.get("requested_fid_count",7000 if manifest["mode"]=="full" else 0)
        evaluate(models,dl,device,out,history,count)
        log.update(status="evaluated",evaluation_seed=42,fid_requested_count=count)
        dump(revision,log)
        manifest.update(dcgan_revision=revision.name,evaluation_seed=42,status="complete")
        dump(out/"manifest.json",manifest)
        return
    if revision.exists():
        p.error("A DCGAN revision already exists; use --evaluate to evaluate the saved checkpoint")
    if not all((out/(k+".pt")).exists() for k in ["MLP","DDPM","LatentDecoder","DCGAN"]):
        p.error("Wait for the initial run to finish training")
    epochs = manifest["epochs"]
    if len(history) != epochs * 8:
        p.error("Initial run has incomplete history")
    model = Adversary(True).to(device)
    optimizers = [torch.optim.Adam(m.parameters(),lr=.0002,betas=(.5,.999)) for m in (model.g,model.d)]
    rows = []
    for e in range(1,epochs+1):
        sync(device)
        start=time.perf_counter()
        train=epoch("DCGAN",model,dl["train"],None,device,optimizers)
        sync(device)
        seconds=time.perf_counter()-start
        val=epoch("DCGAN",model,dl["validation"],None,device)
        rows.append(dict(model="DCGAN",epoch=e,seconds=seconds,train=train,validation=val))
        print(f"DCGAN refresh {e}/{epochs} G={train['loss']:.4f} D={train['discriminator']:.4f}",flush=True)
    fresh=iter(rows)
    replaced=[next(fresh) if row["model"]=="DCGAN" else row for row in history]
    cp=out/"DCGAN.pt"
    torch.save(dict(state=model.state_dict(),epoch=epochs), out/"DCGAN.next.pt")
    dump(out/"history.next.json",replaced)
    os.replace(out/"DCGAN.next.pt",cp)
    os.replace(out/"history.next.json",out/"history.json")
    dump(revision,dict(status="checkpoint_ready",seed=42,epochs=epochs,device=str(device),
         threads=args.threads,lr=.0002,betas=[.5,.999],new_sha256=weight_hash(model),
         previous="dense projection and discriminator head replaced; prototype measurements discarded",
         corrected="fully convolutional generator and discriminator",indices_verified=True))


if __name__=="__main__":
    main()
