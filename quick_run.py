"""Quick end-to-end check (about 8 minutes on an Apple M4 Pro) that writes one CSV and nothing else.

It trains a fresh f-GAN for a few hundred iterations, scores its samples (and, for reference, the
samples of the final trained generator) with FID/KID, and fine-tunes short ResNet-18 classifiers under
the baseline, random over-sampling and f-GAN regimes. Checkpoints and intermediate files go to a
temporary folder that is deleted at the end. Existing runs, results, figures and reports are only read.

    python quick_run.py                    # writes results/quick_run/quick_run_<timestamp>.csv
    python quick_run.py --budget_min 10    # stages that would not fit in the budget are skipped
"""
import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

from fgan.data import CLASSES, NUM_CLASSES, load_cache
from fgan.utils import get_device, seed_everything

HERE = os.path.dirname(os.path.abspath(__file__))


def import_script(name):
    """Import a top-level script by module name, whether its file uses underscores or spaces."""
    for fname in (f"{name}.py", f"{name.replace('_', ' ')}.py"):
        path = os.path.join(HERE, fname)
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget_min", type=float, default=10.0)
    p.add_argument("--gan_iters", type=int, default=300)
    p.add_argument("--cls_steps", type=int, default=400)
    p.add_argument("--n_per_class", type=int, default=100, help="generated images per class for FID/KID")
    p.add_argument("--n_real", type=int, default=2000, help="real training images for FID/KID")
    p.add_argument("--final_ckpt", default="runs/fgan_pearson/generator_ema.pt")
    p.add_argument("--synthetic", default="runs/fgan_pearson/synthetic_balance.npz")
    p.add_argument("--data", default="data/ham10000_128.npz")
    p.add_argument("--out_dir", default="results/quick_run")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    t_start = time.time()
    budget = args.budget_min * 60
    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"quick_run_{stamp}.csv")
    rows = []

    def add(stage, model, metric, value, step="", note=""):
        rows.append({"stage": stage, "model": model, "step": step, "metric": metric,
                     "value": value if isinstance(value, str) else round(float(value), 6), "note": note})

    def fits(estimate_s):
        return time.time() - t_start + estimate_s <= budget

    def save():
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["stage", "model", "step", "metric", "value", "note"])
            w.writeheader()
            w.writerows(rows)

    with tempfile.TemporaryDirectory() as tmp:
        # ---- 1. f-GAN training from scratch (final configuration, short run)
        gan_dir = os.path.join(tmp, "gan")
        est = args.gan_iters * 0.7 + 20
        if fits(est):
            print(f"[1/3] training f-GAN for {args.gan_iters} iterations")
            subprocess.run([sys.executable, "train_fgan.py", "--iters", str(args.gan_iters), "--log_every", "25",
                            "--sample_every", str(10 ** 9), "--ckpt_every", str(10 ** 9), "--out", gan_dir,
                            "--device", args.device], check=True, cwd=HERE)
            with open(os.path.join(gan_dir, "train_log.csv")) as fh:
                for r in csv.DictReader(fh):
                    for k in ("loss_d", "loss_g", "bound", "t_real", "t_fake", "sec"):
                        add("gan_training", "pearson_quick", k, float(r[k]), step=r["iter"])
        else:
            add("gan_training", "pearson_quick", "skipped", "", note=f"needs about {est:.0f} s")
        save()

        # ---- 2. Sample quality: quick generator vs final trained generator, same sample sizes
        est = 90
        if fits(est):
            print("[2/3] FID/KID of the quick and final generators")
            gs = import_script("generate_synthetic")
            ev = import_script("evaluate_gan")
            dev = get_device(args.device)
            inc = ev.InceptionFeatures().to(dev)
            real, labels = load_cache(args.data, "train")
            rng = np.random.default_rng(0)
            idx = rng.choice(len(real), min(args.n_real, len(real)), replace=False)
            real_feat = ev.features(inc, real[idx], dev)
            models = [("pearson_quick", os.path.join(gan_dir, "generator_ema.pt"), f"{args.gan_iters} it"),
                      ("pearson_final", args.final_ckpt, "trained model, 8000 it")]
            for name, path, note in models:
                if not os.path.exists(path):
                    add("sample_quality", name, "skipped", "", note="checkpoint not found")
                    continue
                seed_everything(0)
                G, z_dim = gs.load_generator(path, dev)
                gen = np.concatenate([gs.sample(G, z_dim, c, args.n_per_class, dev) for c in range(NUM_CLASSES)])
                gen_feat = ev.features(inc, gen, dev)
                add("sample_quality", name, "fid", ev.fid(real_feat, gen_feat),
                    note=f"{note}; {len(gen)} generated vs {len(idx)} real")
                add("sample_quality", name, "kid_x1000", 1000 * ev.kid(real_feat, gen_feat)[0], note=note)
        else:
            add("sample_quality", "", "skipped", "", note="not enough time left in the budget")
        save()

        # ---- 3. Short downstream classifiers on the real test split
        cls_dir = os.path.join(tmp, "cls")
        for regime in ("baseline", "ros", "fgan"):
            est = args.cls_steps * 0.13 + 25
            if not fits(est):
                add("classifier", regime, "skipped", "", note="not enough time left in the budget")
                continue
            print(f"[3/3] ResNet-18, {regime}, {args.cls_steps} steps")
            subprocess.run([sys.executable, "train_classifier.py", "--regime", regime, "--steps", str(args.cls_steps),
                            "--synthetic", args.synthetic, "--out", cls_dir, "--seed", "0", "--device", args.device],
                           check=True, cwd=HERE)
            with open(os.path.join(cls_dir, f"{regime}_seed0.json")) as fh:
                res = json.load(fh)
            note = f"{args.cls_steps} steps, seed 0"
            for k in ("accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall",
                      "auc_ovr_macro"):
                add("classifier", regime, k, res[k], step=args.cls_steps, note=note)
            for c in CLASSES:
                add("classifier", regime, f"recall_{c}", res["per_class_recall"][c], step=args.cls_steps, note=note)
            save()

    add("run", "", "total_seconds", time.time() - t_start)
    save()
    print(f"wrote {out_csv} ({len(rows)} rows, {time.time() - t_start:.0f} s)")


if __name__ == "__main__":
    main()
