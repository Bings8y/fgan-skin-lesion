"""Decode HAM10000 once into a 128x128 uint8 cache with a lesion-grouped 80/20 train/test split.

Expects data/raw/HAM10000_metadata.csv and the two HAM10000_images_part_*.zip archives
(Harvard Dataverse doi:10.7910/DVN/DBW86T).
"""
import argparse

import numpy as np

from fgan.data import CLASSES, build_cache, class_counts

p = argparse.ArgumentParser()
p.add_argument("--raw", default="data/raw")
p.add_argument("--out", default="data/ham10000_128.npz")
p.add_argument("--size", type=int, default=128)
args = p.parse_args()

images, labels, split = build_cache(args.raw, args.out, size=args.size)
print(f"{len(images)} images cached to {args.out}")
for name, m in (("all", np.ones_like(split, bool)), ("train", split == 0), ("test", split == 1)):
    counts = class_counts(labels[m])
    print(f"{name:5s} n={m.sum():5d}  " + "  ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))
