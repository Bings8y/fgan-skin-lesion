"""Sample class-conditional lesions from the trained EMA generator.

--mode balance  tops every class in the training split up to --target images (synthetic minority set)
--mode fixed    draws --per_class images for every class (used for FID/KID evaluation)
"""
import argparse

import numpy as np
import torch

from fgan.data import CLASSES, NUM_CLASSES, class_counts, load_cache
from fgan.models import Generator
from fgan.utils import get_device, seed_everything, to_uint8


def load_generator(path, dev):
    ck = torch.load(path, map_location=dev)
    G = Generator(NUM_CLASSES, ck["args"]["z_dim"]).to(dev).eval()
    G.load_state_dict(ck["G_ema"])
    return G, ck["args"]["z_dim"]


@torch.no_grad()
def sample(G, z_dim, cls, n, dev, batch=128):
    out = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        z = torch.randn(b, z_dim, device=dev)
        c = torch.full((b,), cls, dtype=torch.long, device=dev)
        out.append(to_uint8(G(z, c)))
    return np.concatenate(out) if out else np.zeros((0, 128, 128, 3), np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="runs/fgan_pearson/checkpoint.pt")
    p.add_argument("--data", default="data/ham10000_128.npz")
    p.add_argument("--mode", default="balance", choices=["balance", "fixed"])
    p.add_argument("--target", type=int, default=2000)
    p.add_argument("--per_class", type=int, default=1000)
    p.add_argument("--out", default="runs/fgan_pearson/synthetic_balance.npz")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    seed_everything(args.seed)
    dev = get_device(args.device)
    G, z_dim = load_generator(args.ckpt, dev)
    if args.mode == "balance":
        _, labels = load_cache(args.data, "train")
        need = np.maximum(args.target - class_counts(labels), 0)
    else:
        need = np.full(NUM_CLASSES, args.per_class)

    images, labels = [], []
    for c in range(NUM_CLASSES):
        images.append(sample(G, z_dim, c, int(need[c]), dev))
        labels.append(np.full(int(need[c]), c, dtype=np.int64))
        print(f"{CLASSES[c]:6s} +{need[c]}")
    np.savez(args.out, images=np.concatenate(images), labels=np.concatenate(labels))
    print(f"saved {sum(need)} synthetic images to {args.out}")


if __name__ == "__main__":
    main()
