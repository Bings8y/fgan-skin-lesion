"""Sample-quality metrics for trained generators: FID and KID in InceptionV3 pool features.

Features come from torchvision's ImageNet InceptionV3 (2048-d pool layer), so absolute values
are comparable across runs in this project but not to numbers from the reference TF FID code.

Example:
    python evaluate_gan.py --ckpt pearson=runs/fgan_pearson/checkpoint.pt --out results/gan_metrics.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torchvision.models import Inception_V3_Weights, inception_v3

from fgan.data import CLASSES, NUM_CLASSES, class_counts, load_cache
from fgan.utils import get_device, seed_everything
from generate_synthetic import load_generator, sample

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class InceptionFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        net.fc = torch.nn.Identity()
        self.net = net.eval()

    @torch.no_grad()
    def forward(self, x_uint8):
        x = x_uint8.permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        return self.net(x)


@torch.no_grad()
def features(model, images, dev, batch=100):
    out = []
    for i in range(0, len(images), batch):
        out.append(model(torch.from_numpy(images[i:i + batch]).to(dev)).cpu().double().numpy())
    return np.concatenate(out)


def fid(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def kid(a, b, n_subsets=50, subset=100, seed=0):
    """Unbiased MMD^2 with the cubic polynomial kernel (Binkowski et al., 2018); robust for small n."""
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    m = min(subset, len(a), len(b))
    vals = []
    for _ in range(n_subsets):
        x = a[rng.choice(len(a), m, replace=False)]
        y = b[rng.choice(len(b), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="*", default=[], help="name=path pairs")
    p.add_argument("--reference", action="store_true",
                   help="also score the real test split against the real training split (a floor for FID/KID)")
    p.add_argument("--data", default="data/ham10000_128.npz")
    p.add_argument("--n_overall", type=int, default=5000)
    p.add_argument("--min_per_class", type=int, default=500)
    p.add_argument("--out", default="results/gan_metrics.json")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    dev = get_device(args.device)
    inc = InceptionFeatures().to(dev)
    real, labels = load_cache(args.data, "train")
    real_feat = features(inc, real, dev)
    counts = class_counts(labels)

    results = {}
    if os.path.exists(args.out):
        with open(args.out) as fh:
            results = json.load(fh)
    if args.reference:
        test, test_labels = load_cache(args.data, "test")
        test_feat = features(inc, test, dev)
        res = {"fid_overall": fid(real_feat, test_feat), "kid_overall": kid(real_feat, test_feat, subset=1000)[0],
               "per_class": {}}
        for c in range(NUM_CLASSES):
            rf, tf = real_feat[labels == c], test_feat[test_labels == c]
            res["per_class"][CLASSES[c]] = {"n_real": int(counts[c]), "fid": fid(rf, tf), "kid": kid(rf, tf)[0]}
        results["real_test_reference"] = res
        print(f"real test vs real train: FID {res['fid_overall']:.2f}  KIDx1e3 {1e3 * res['kid_overall']:.2f}")

    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        seed_everything(0)
        G, z_dim = load_generator(path, dev)
        res = {"per_class": {}}
        # Overall: generated set follows the training-set class prior.
        n_c = np.round(counts / counts.sum() * args.n_overall).astype(int)
        gen_all = np.concatenate([sample(G, z_dim, c, int(n_c[c]), dev) for c in range(NUM_CLASSES)])
        gen_feat_all = features(inc, gen_all, dev)
        res["fid_overall"] = fid(real_feat, gen_feat_all)
        res["kid_overall"] = kid(real_feat, gen_feat_all, subset=1000)[0]
        for c in range(NUM_CLASSES):
            rf = real_feat[labels == c]
            gf = features(inc, sample(G, z_dim, c, max(int(counts[c]), args.min_per_class), dev), dev)
            k_mean, k_std = kid(rf, gf)
            res["per_class"][CLASSES[c]] = {"n_real": int(counts[c]), "fid": fid(rf, gf),
                                            "kid": k_mean, "kid_std": k_std}
        results[name] = res
        print(name, f"FID {res['fid_overall']:.2f}  KIDx1e3 {1e3 * res['kid_overall']:.2f}")
        for cname, r in res["per_class"].items():
            print(f"   {cname:6s} FID {r['fid']:7.2f}  KIDx1e3 {1e3 * r['kid']:6.2f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
