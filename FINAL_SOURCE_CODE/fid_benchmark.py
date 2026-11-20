"""Inception-v3 pool3 FID on matched test and generated image counts."""
import math
import os
import time

import torch

from evaluate import generate
from full_benchmark import write_json


def image_bytes(x):
    return (x.clamp(0,1)*255).round().to(torch.uint8).repeat(1,3,1,1)


@torch.inference_mode()
def score_models(models,real,count,device,out,results,classifier=None):
    required=("VAE","GAN","DCGAN","DDPM")
    if set(required)-set(models):
        raise ValueError("Provide VAE, GAN, DCGAN and DDPM for a matched FID comparison")
    results.update(fid_requested_count=count,fid_feature_dimension=2048,fid_extractor="torch-fidelity Inception-v3",
                   fid_preprocessing="round grayscale [0,1] to uint8, repeat channel RGB, extractor resizes to 299",
                   fid_reference="first N fixed-order held-out test images",fid_seed=42,
                   fid_limitation="Finite-sample biased; ImageNet representation/domain mismatch. "
                                  "Estimates below 2049 samples have singular sample covariance.")
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        opts={}
        weights=os.environ.get("FID_WEIGHTS")
        if weights:
            opts["feature_extractor_weights_path"]=weights
        metric=FrechetInceptionDistance(feature=2048,normalize=False,reset_real_features=False,**opts).to(device)
    except Exception as e:
        results["fid_status"]=f"Extractor unavailable: {type(e).__name__}: {e}"
        for name in required:
            results["models"][name].update(fid=None,fid_real_count=0,fid_fake_count=0)
        write_json(out/"metrics.json",results)
        print(results["fid_status"],flush=True)
        return
    for x in real[:count].split(64):
        metric.update(image_bytes(x),real=True)
    if classifier is not None:
        results["predicted_class_histogram_protocol"] = (
            "Frozen Fashion-MNIST MLP argmax; diagnostic only, no ground-truth labels "
            "for generated samples; classifier accuracy on synthetic data unknown."
        )
        labels=torch.cat([classifier(x).argmax(1) for x in real[:count].split(64)])
        results["real_predicted_class_histogram"]=torch.bincount(labels,minlength=10).cpu().tolist()
    results["fid_status"]="running"
    write_json(out/"metrics.json",results)
    for name in required:
        torch.manual_seed(42)
        metric.reset()
        start=time.perf_counter()
        predicted=[]
        for i in range(0,count,64):
            fake=generate(models,name,min(64,count-i),device)
            metric.update(image_bytes(fake),real=False)
            if classifier is not None:
                predicted.append(classifier(fake).argmax(1))
            if (i//64)%8==0:
                print(f"FID {name} generated {min(count,i+64)}/{count}",flush=True)
        value=metric.compute().item()
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite FID for {name}")
        results["models"][name].update(fid=value,fid_real_count=count,fid_fake_count=count,
                                       fid_wall_seconds=time.perf_counter()-start)
        if predicted:
            histogram=torch.bincount(torch.cat(predicted),minlength=10).cpu().tolist()
            results["models"][name]["predicted_class_histogram"]=histogram
            results["models"][name]["predicted_classes_at_least_one_percent"]=sum(
                100*frequency>=count for frequency in histogram
            )
        write_json(out/"metrics.json",results)
        print(f"FID {name} = {value:.5f} (n={count})",flush=True)
    results["fid_status"]="computed"
    write_json(out/"metrics.json",results)
