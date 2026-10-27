"""Downstream diagnostic validation: ResNet-18 trained under different rebalancing regimes.

Regimes (all trained for the same number of optimiser steps and evaluated on the same
untouched real test split):
    baseline  original imbalanced training images
    ros       random over-sampling: minority images duplicated up to --target per class
    fgan      real images + f-GAN synthetic minority images up to --target per class

Example:
    python train_classifier.py --regime fgan --synthetic runs/fgan_pearson/synthetic_balance.npz --seed 0
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torchvision.models import ResNet18_Weights, resnet18

from fgan.data import CLASSES, NUM_CLASSES, class_counts, gpu_augment, load_cache
from fgan.utils import get_device, seed_everything

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_training_set(regime, data, synthetic, target, rng):
    images, labels = load_cache(data, "train")
    if regime == "ros":
        counts = class_counts(labels)
        extra = [rng.choice(np.flatnonzero(labels == c), max(target - counts[c], 0), replace=True)
                 for c in range(NUM_CLASSES)]
        extra = np.concatenate(extra)
        images, labels = np.concatenate([images, images[extra]]), np.concatenate([labels, labels[extra]])
    elif regime == "fgan":
        syn = np.load(synthetic)
        images = np.concatenate([images, syn["images"]])
        labels = np.concatenate([labels, syn["labels"]])
    return images, labels


def normalise(x_m11):
    x = (x_m11 + 1) / 2
    return (x - MEAN.to(x.device)) / STD.to(x.device)


@torch.no_grad()
def predict(model, images, dev, batch=256):
    model.eval()
    probs = []
    for i in range(0, len(images), batch):
        x = torch.from_numpy(images[i:i + batch]).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
        logits = model(normalise(x)) + model(normalise(x.flip(3)))  # flip test-time augmentation
        probs.append(torch.softmax(logits / 2, dim=1).cpu().numpy())
    return np.concatenate(probs)


def metrics(y, prob):
    pred = prob.argmax(1)
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro")),
        "auc_ovr_macro": float(roc_auc_score(y, prob, multi_class="ovr", average="macro")),
        "per_class_recall": dict(zip(CLASSES, recall_score(y, pred, average=None).round(4).tolist())),
        "per_class_f1": dict(zip(CLASSES, f1_score(y, pred, average=None).round(4).tolist())),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(NUM_CLASSES))).tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--regime", required=True, choices=["baseline", "ros", "fgan"])
    p.add_argument("--data", default="data/ham10000_128.npz")
    p.add_argument("--synthetic", default="runs/fgan_pearson/synthetic_balance.npz")
    p.add_argument("--target", type=int, default=2000)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="label for the result files (default: the regime)")
    p.add_argument("--out", default="results/classifier")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = get_device(args.device)
    tr_x, tr_y = build_training_set(args.regime, args.data, args.synthetic, args.target, rng)
    te_x, te_y = load_cache(args.data, "test")
    print(f"[{args.regime} seed {args.seed}] train n={len(tr_x)} counts={class_counts(tr_y).tolist()}  "
          f"test n={len(te_x)}")

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss()

    x_all = torch.from_numpy(tr_x).permute(0, 3, 1, 2).contiguous().to(dev)
    y_all = torch.from_numpy(tr_y).to(dev)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    perm = torch.randperm(len(tr_x), generator=g)
    ptr = 0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        if ptr + args.batch > len(perm):  # reshuffle each epoch
            perm, ptr = torch.randperm(len(tr_x), generator=g), 0
        idx = perm[ptr:ptr + args.batch].to(dev)
        ptr += args.batch
        x = gpu_augment(x_all[idx].float() / 127.5 - 1)
        loss = loss_fn(model(normalise(x)), y_all[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"  step {step}  loss {loss.item():.4f}  {time.time() - t0:.0f}s", flush=True)

    prob = predict(model, te_x, dev)
    name = args.name or args.regime
    res = {"regime": name, "seed": args.seed, "train_counts": class_counts(tr_y).tolist(),
           "steps": args.steps, **metrics(te_y, prob)}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{name}_seed{args.seed}.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    np.save(os.path.join(args.out, f"{name}_seed{args.seed}_probs.npy"), prob)
    print(f"  bal_acc {res['balanced_accuracy']:.4f}  macro_f1 {res['macro_f1']:.4f}  "
          f"auc {res['auc_ovr_macro']:.4f}  acc {res['accuracy']:.4f}")
    print("  recall", res["per_class_recall"])


if __name__ == "__main__":
    main()
