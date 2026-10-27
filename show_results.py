"""Demo: prints the trained f-GAN, the headline results, and opens windows with generated lesions and figures.

    python show_results.py
"""
import json
import os

import matplotlib.pyplot as plt
import torch

from fgan.data import CLASSES
from generate_synthetic import load_generator

ROOT = os.path.dirname(os.path.abspath(__file__))
NAMES = {"akiec": "Actinic keratosis", "bcc": "Basal cell carcinoma", "bkl": "Benign keratosis",
         "df": "Dermatofibroma", "mel": "Melanoma", "nv": "Melanocytic nevus", "vasc": "Vascular lesion"}


def path(*parts):
    return os.path.join(ROOT, *parts)


def main():
    G, z_dim = load_generator(path("runs", "fgan_pearson", "checkpoint.pt"), "cpu")
    n_params = sum(p.numel() for p in G.parameters()) / 1e6

    print("=" * 64)
    print(" f-GAN (Pearson chi-squared) for imbalanced skin-lesion data")
    print("=" * 64)
    print(f"\nTrained generator: {n_params:.1f}M parameters, latent size {z_dim}, 7 classes, 128x128 RGB")
    print("\nLayers:")
    for name, module in G.named_children():
        print(f"  {name:<10} {module.__class__.__name__}")

    gan = json.load(open(path("results", "gan_metrics.json")))
    print("\nImage quality (FID, lower is better)")
    for key, label in [("real_test_reference", "Real vs real (best possible)"),
                       ("js_3k", "Jensen-Shannon f-GAN, 3k iterations"),
                       ("pearson_3k", "Pearson f-GAN, 3k iterations"),
                       ("pearson_8k", "Pearson f-GAN, 8k iterations (final)")]:
        print(f"  {label:<40} {gan[key]['fid_overall']:7.1f}")

    cls = json.load(open(path("results", "classifier_summary.json")))
    print("\nResNet-18 skin-lesion classifier on real test images (mean of 3 seeds)")
    print(f"  {'Training data':<32} {'Bal. acc':>9} {'Macro F1':>9} {'AUC':>7}")
    for key, label in [("baseline", "Real only (imbalanced)"), ("ros", "Random over-sampling"),
                       ("fgan", "+ f-GAN, 2000 per class"), ("fgan1000", "+ f-GAN, 1000 per class")]:
        r = cls[key]
        print(f"  {label:<32} {r['balanced_accuracy']['mean']:9.3f} {r['macro_f1']['mean']:9.3f} "
              f"{r['auc_ovr_macro']['mean']:7.3f}")

    print("\nGenerating new lesion images with the trained model ...")
    per_class = 6
    torch.manual_seed(int.from_bytes(os.urandom(4), "little"))
    labels = torch.arange(len(CLASSES)).repeat_interleave(per_class)
    with torch.no_grad():
        images = (G(torch.randn(len(labels), z_dim), labels) * 0.5 + 0.5).clamp(0, 1)

    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(per_class * 1.6, len(CLASSES) * 1.6))
    fig.suptitle("New synthetic skin lesions from the trained f-GAN (one row per class)")
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].permute(1, 2, 0).numpy())
        ax.set_xticks([])
        ax.set_yticks([])
        if i % per_class == 0:
            ax.set_ylabel(NAMES[CLASSES[i // per_class]], rotation=0, ha="right", va="center", fontsize=9)
    fig.tight_layout()

    for fname, title in [("real_vs_synthetic.png", "Real vs synthetic lesions"),
                         ("training_curves.png", "f-GAN training curves"),
                         ("per_class_recall.png", "Classifier recall per class")]:
        f = path("results", "figures", fname)
        if os.path.exists(f):
            fig = plt.figure(figsize=(10, 7))
            fig.canvas.manager.set_window_title(title)
            plt.imshow(plt.imread(f))
            plt.axis("off")
            plt.tight_layout()

    print("Opening result windows (close them to finish).")
    plt.show()


if __name__ == "__main__":
    main()
