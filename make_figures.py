"""Aggregate results and render every figure used in the report into results/figures/."""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from fgan.data import CLASSES, NUM_CLASSES, class_counts, load_cache

DATA = "data/ham10000_128.npz"
RUN = "runs/fgan_pearson"
RES = "results"
FIG = os.path.join(RES, "figures")
REGIMES = [("baseline", "Baseline (imbalanced)"), ("ros", "Random over-sampling"),
           ("fgan", "f-GAN balanced (2,000/class)"), ("fgan1000", "f-GAN balanced (1,000/class)")]
COLORS = {"baseline": "#8c8c8c", "ros": "#e69f00", "fgan": "#0072b2", "fgan1000": "#56b4e9",
          "js": "#d55e00", "pearson": "#0072b2"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})


def fig_class_distribution():
    _, y = load_cache(DATA, "train")
    before = class_counts(y)
    syn = np.load(os.path.join(RUN, "synthetic_balance.npz"))["labels"]
    added = class_counts(syn)
    x = np.arange(NUM_CLASSES)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(x, before, color="#8c8c8c", label="Real training images")
    ax.bar(x, added, bottom=before, color="#0072b2", label="f-GAN synthetic images")
    for i in x:
        ax.text(i, before[i] + added[i] + 60, f"{before[i]}" + (f"+{added[i]}" if added[i] else ""),
                ha="center", fontsize=8)
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("Images")
    ax.set_title("Training-set class distribution before and after f-GAN balancing")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(FIG, "class_distribution.png"))
    plt.close(fig)


def fig_training_curves():
    runs = [("pearson", RUN, "Pearson χ²"), ("js", "runs/fgan_js", "Jensen-Shannon")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for key, path, label in runs:
        f = os.path.join(path, "train_log.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        sm = lambda s: s.rolling(10, min_periods=1).mean()
        axes[0].plot(d["iter"], sm(d["loss_d"]), color=COLORS[key], label=label)
        axes[1].plot(d["iter"], sm(d["loss_g"]), color=COLORS[key], label=label)
        axes[2].plot(d["iter"], sm(d["t_real"]), color=COLORS[key], label=f"{label}: T(real)")
        axes[2].plot(d["iter"], sm(d["t_fake"]), color=COLORS[key], ls="--", label=f"{label}: T(fake)")
    for ax, t in zip(axes, ["Critic loss $L_D$", "Generator loss $L_G$", "Mean critic output T"]):
        ax.set_title(t)
        ax.set_xlabel("Iteration")
    axes[0].legend(frameon=False)
    axes[2].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "training_curves.png"))
    plt.close(fig)


def fig_samples(n=6, seed=0):
    real, y = load_cache(DATA, "train")
    syn = np.load(os.path.join(RUN, "synthetic_fixed.npz"))
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(NUM_CLASSES, 2 * n, figsize=(2 * n * 0.95, NUM_CLASSES * 1.0))
    for c in range(NUM_CLASSES):
        r = real[rng.choice(np.flatnonzero(y == c), n, replace=False)]
        s = syn["images"][rng.choice(np.flatnonzero(syn["labels"] == c), n, replace=False)]
        for j, img in enumerate(list(r) + list(s)):
            ax = axes[c, j]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[c, 0].set_ylabel(CLASSES[c], rotation=0, ha="right", va="center", fontsize=10)
    axes[0, n // 2].set_title("Real (HAM10000)", fontsize=11, loc="left")
    axes[0, n + n // 2].set_title("Synthetic (f-GAN, Pearson χ²)", fontsize=11, loc="left")
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    fig.savefig(os.path.join(FIG, "real_vs_synthetic.png"))
    plt.close(fig)


def fig_nearest_neighbours(per_class=1, k=4, seed=1):
    """Each synthetic sample next to its k nearest real training images (L2 on 32x32 thumbnails)."""
    real, y = load_cache(DATA, "train")
    syn = np.load(os.path.join(RUN, "synthetic_fixed.npz"))
    thumb = lambda a: a.reshape(len(a), 32, 4, 32, 4, 3).mean((2, 4)).reshape(len(a), -1)
    rng = np.random.default_rng(seed)
    rows = []
    for c in [CLASSES.index(n) for n in ("mel", "bcc", "df", "vasc")]:
        idx = np.flatnonzero(syn["labels"] == c)
        for s in rng.choice(idx, per_class, replace=False):
            pool = np.flatnonzero(y == c)
            d = ((thumb(real[pool]) - thumb(syn["images"][s:s + 1])) ** 2).sum(1)
            rows.append((CLASSES[c], syn["images"][s], real[pool[np.argsort(d)[:k]]]))
    fig, axes = plt.subplots(len(rows), k + 1, figsize=((k + 1) * 1.3, len(rows) * 1.3))
    for i, (name, s, nn_imgs) in enumerate(rows):
        for j, img in enumerate([s, *nn_imgs]):
            axes[i, j].imshow(img)
            axes[i, j].axis("off")
        axes[i, 0].text(-8, 64, name, ha="right", va="center")
    axes[0, 0].set_title("Synthetic", fontsize=9)
    for j in range(1, k + 1):
        axes[0, j].set_title(f"Real NN {j}", fontsize=9)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(os.path.join(FIG, "nearest_neighbours.png"))
    plt.close(fig)


def load_classifier_results():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "classifier", "*_seed*.json"))):
        with open(f) as fh:
            rows.append(json.load(fh))
    return rows


def summarise_classifiers(rows):
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall", "auc_ovr_macro"]
    out = {}
    for reg, _ in REGIMES:
        rs = [r for r in rows if r["regime"] == reg]
        if not rs:
            continue
        out[reg] = {"n_seeds": len(rs)}
        for k in keys:
            v = np.array([r[k] for r in rs])
            out[reg][k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
        for k in ("per_class_recall", "per_class_f1"):
            v = np.array([[r[k][c] for c in CLASSES] for r in rs])
            out[reg][k] = {c: {"mean": float(m), "std": float(s)}
                           for c, m, s in zip(CLASSES, v.mean(0), v.std(0, ddof=1) if len(v) > 1 else 0 * v[0])}
        out[reg]["confusion_matrix_sum"] = np.sum([r["confusion_matrix"] for r in rs], axis=0).tolist()
    with open(os.path.join(RES, "classifier_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return out


def fig_per_class_recall(summary):
    x = np.arange(NUM_CLASSES)
    regs = [r for r in REGIMES if r[0] in summary]
    w = 0.8 / len(regs)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, (reg, label) in enumerate(regs):
        m = [summary[reg]["per_class_recall"][c]["mean"] for c in CLASSES]
        s = [summary[reg]["per_class_recall"][c]["std"] for c in CLASSES]
        ax.bar(x + (i - (len(regs) - 1) / 2) * w, m, w, yerr=s, capsize=2, color=COLORS[reg], label=label)
    ax.set_xticks(x, CLASSES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Sensitivity (recall)")
    ax.set_title("Per-class sensitivity on the real test split (mean ± s.d. over seeds)")
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
    fig.savefig(os.path.join(FIG, "per_class_recall.png"))
    plt.close(fig)


def fig_confusion(summary):
    regs = [r for r in REGIMES if r[0] in summary]
    ncol = 2
    nrow = (len(regs) + 1) // 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.4, 3.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(regs):]:
        ax.axis("off")
    for ax, (reg, label) in zip(axes, regs):
        cm = np.array(summary[reg]["confusion_matrix_sum"], dtype=float)
        cm = cm / cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > 0.6 else "black")
        ax.set_xticks(range(NUM_CLASSES), CLASSES, rotation=45)
        ax.set_yticks(range(NUM_CLASSES), CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_title(label, fontsize=10)
    for ax in axes[::ncol]:
        ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "confusion_matrices.png"))
    plt.close(fig)


def fig_conjugates():
    """Generator gradient weight d f*(T)/dT for Pearson vs JS, illustrating the non-saturating behaviour."""
    v = np.linspace(-6, 6, 400)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    t_p = v
    axes[0].plot(v, 0.25 * t_p ** 2 + t_p, color=COLORS["pearson"], label="Pearson χ²: f*(T) = T²/4 + T")
    # JS: T = log 2 - softplus(-v), so f*(T) = -log(2 - e^T) = softplus(v) - log 2
    axes[0].plot(v, np.log1p(np.exp(v)) - np.log(2), color=COLORS["js"], label="JS: f*(T) = −log(2 − eᵀ)")
    axes[0].set_xlabel("Raw critic output v")
    axes[0].set_title("Conjugate f*(g_f(v)) seen by the generator")
    axes[0].legend(frameon=False, fontsize=8)
    # d/dv of f*(g_f(v)): Pearson -> v/2 + 1 ; JS -> sigmoid(v), which vanishes as v -> -inf
    axes[1].plot(v, v / 2 + 1, color=COLORS["pearson"], label="Pearson χ²: T/2 + 1")
    axes[1].plot(v, 1 / (1 + np.exp(-v)), color=COLORS["js"], label="JS: σ(v)")
    axes[1].axhline(0, color="#bbbbbb", lw=0.8)
    axes[1].set_xlabel("Raw critic output v")
    axes[1].set_title("Generator gradient weight ∂f*/∂v")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "conjugate_gradients.png"))
    plt.close(fig)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 110); ax.set_ylim(-4, 47); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=8.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec="#333333", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, color="#333333",
                                     shrinkA=0, shrinkB=0))

    def label(x, y, text, **kw):
        ax.text(x, y, text, fontsize=7, color="#444444", **kw)

    data_c, gan_c, syn_c, eval_c = "#e8eef7", "#dbeedd", "#fbe9d0", "#f3e1ee"
    # stage panels
    for x, w, title in [(0, 20, "1. Data"), (23.5, 25, "2. f-GAN training"), (52, 23, "3. Synthesis"),
                        (78.5, 31, "4. Downstream validation")]:
        ax.add_patch(FancyBboxPatch((x, -1), w, 45, boxstyle="round,pad=0,rounding_size=1.5",
                                    fc="#fafafa", ec="#cccccc", lw=0.8, ls="--"))
        ax.text(x + w / 2, 45.2, title, ha="center", fontsize=9.5, weight="bold")

    box(1.5, 31, 17, 10, "HAM10000\n10,015 dermoscopy\nimages, 7 classes", data_c)
    box(1.5, 17, 17, 9.5, "Preprocessing\ncentre crop, 128×128\nscale to [−1, 1]", data_c)
    box(1.5, 2.5, 17, 10, "Lesion-grouped split\ntrain 80% / test 20%\n(no lesion leakage)", data_c)
    arrow(10, 30.5, 10, 27.2); arrow(10, 16.5, 10, 13.2)

    box(25, 31, 22, 10, "Generator G(z, c)\nz ~ N(0, I) ⊕ Embed(c) → 512×4×4\n5 × ConvTranspose + CBN + ReLU\nTanh → 128×128×3", gan_c, 6.5)
    box(25, 16, 22, 10.5, "Variational critic T(x, c)\n5 × SN-Conv 4×4 s2 (+3×3 at low res)\nminibatch std-dev, sum-pool ψ(x)\nwᵀψ + ⟨ψ, v_c⟩, no sigmoid", gan_c, 6.5)
    box(25, 2, 22, 9.5, "Pearson χ² to the mixture\nL_D = −E_P[T − T²/4] + E_Q[T²/4 + T]\nL_G = −E_Q[T] + mode seeking\nTTUR Adam (β₁ = 0), EMA of G", gan_c, 6.4)
    arrow(36, 30.5, 36, 27.2); label(37, 28.2, "G(z, c)")
    arrow(36, 15.5, 36, 12.2); label(37, 13.2, "T(real), T(fake)")
    arrow(19.3, 8, 24.3, 19); label(19.6, 14.5, "real\n(x, c)", ha="left")

    box(53.5, 28, 20, 13, "Synthetic balancing\nEMA generator tops up\neach class to 2,000\n(or 1,000) images", syn_c)
    box(53.5, 4, 20, 13, "Sample-quality checks\nFID / KID (InceptionV3)\nnearest-neighbour\nmemorisation check", syn_c)
    arrow(47.8, 37, 52.8, 35)
    arrow(47.8, 34, 52.8, 13)

    box(80, 28, 28, 13, "ResNet-18, equal training budget\nA: baseline (imbalanced)\nB: random over-sampling\nC: real + f-GAN (2,000/class)\nD: real + f-GAN (1,000/class)", eval_c)
    box(80, 4, 28, 13, "Evaluation on the untouched\nreal test split\nBalanced accuracy · Macro F1\nPrecision · Sensitivity · AUC-ROC", eval_c)
    arrow(74.3, 34.5, 79.3, 34.5)
    arrow(94, 27.5, 94, 17.7)
    # test split routed along the bottom so it crosses no boxes
    ax.plot([10, 10, 101, 101], [2, -2.6, -2.6, 0], color="#333333", lw=1)
    arrow(101, 0, 101, 3.3)
    label(55, -2.2, "held-out real test images (never used to train the GAN or the classifiers)", ha="center")
    fig.savefig(os.path.join(FIG, "architecture.png"))
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_architecture()
    fig_conjugates()
    fig_class_distribution()
    fig_training_curves()
    fig_samples()
    fig_nearest_neighbours()
    rows = load_classifier_results()
    if rows:
        summary = summarise_classifiers(rows)
        fig_per_class_recall(summary)
        fig_confusion(summary)
        for reg, s in summary.items():
            print(f"{reg:9s} n={s['n_seeds']}  " + "  ".join(
                f"{k}={s[k]['mean']:.4f}±{s[k]['std']:.4f}"
                for k in ("accuracy", "balanced_accuracy", "macro_f1", "auc_ovr_macro")))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
