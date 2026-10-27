# f-GAN for Imbalanced Dermatological Datasets

Class-conditional **f-GAN** trained with the **Pearson χ²** variational divergence bound to synthesise
128×128 dermoscopic lesions for all seven HAM10000 diagnostic classes. The synthetic images rebalance
the minority classes, and their effect is measured with ResNet-18 classifiers on an untouched real test split.

## Layout

| Path | Purpose |
|---|---|
| `fgan/data.py` | HAM10000 cache (centre crop → 128×128), lesion-grouped 80/20 split, GPU augmentation |
| `fgan/models.py` | Generator with Conditional BatchNorm, spectrally normalised projection critic |
| `fgan/losses.py` | f-divergence objectives (Pearson χ², JS, KL, reverse KL) via Fenchel conjugates |
| `prepare_data.py` | builds `data/ham10000_128.npz` from the Dataverse archives |
| `train_fgan.py` | f-GAN training (TTUR Adam, β₁ = 0, EMA generator, samples and checkpoints) |
| `generate_synthetic.py` | tops minority classes up to a target count or draws a fixed per-class set |
| `evaluate_gan.py` | FID / KID (overall and per class) in InceptionV3 feature space |
| `train_classifier.py` | ResNet-18 under `baseline`, `ros` (random over-sampling) or `fgan` regimes |
| `make_figures.py` | figures and the aggregated `results/classifier_summary.json` |
| `report/` | builds the final e-report from the institutional Word template (`content.txt` is the text) |
| `journal/` | conference paper: `paper.txt` is the text, `build_conference_docx.py` builds the two-column .docx |
| `run_all.sh` | the full pipeline end to end |

## Reproduce

```bash
pip install -r requirements.txt
# HAM10000 (Tschandl et al., 2018) from Harvard Dataverse, doi:10.7910/DVN/DBW86T:
#   data/raw/HAM10000_metadata.csv, data/raw/part1.zip, data/raw/part2.zip
bash run_all.sh
```

Training runs on CUDA, Apple MPS or CPU (`--device`). On an Apple M4 Pro one GAN iteration
(batch 64, 128×128) takes about 0.65 s.

## Results (seed means ± s.d., real test split)

| | FID | Balanced acc. | Macro F1 | AUC |
|---|---|---|---|---|
| Real test vs real train (floor) | 9.7 | | | |
| Jensen–Shannon f-GAN, 3k it | 186.5 | | | |
| Pearson χ² f-GAN, 3k it | 163.7 | | | |
| **Pearson χ² f-GAN, 8k it (final)** | **124.7** | | | |
| ResNet-18 baseline (imbalanced) | | 0.606 ± 0.015 | 0.662 ± 0.013 | 0.953 |
| + random over-sampling | | **0.624 ± 0.013** | **0.677 ± 0.012** | 0.957 |
| + f-GAN, 2,000/class | | 0.601 ± 0.036 | 0.642 ± 0.031 | 0.957 |
| + f-GAN, 1,000/class | | 0.594 ± 0.018 | 0.640 ± 0.008 | 0.954 |

The generator trains stably and produces varied, non-memorised lesions, but at 128×128 and 8k
iterations the synthetic data does not beat random over-sampling downstream. See the report, Chapter 5.

## Deviations from the proposal (and why)

The proposal's plain Pearson χ² objective was unstable (see `results/pilots/README.md`):

* **Critic bound.** `--drift 0.25` adds ¼·E_P[T²] to the critic loss. The critic then estimates
  2·χ²(P ‖ (P+Q)/2), which is bounded in [0, 2], with optimal T ∈ [−2, 2], instead of an unbounded χ²(P ‖ Q).
* **Generator loss.** Non-saturating −E[T] (Nowozin et al. §3.2) instead of the saddle −E[f*(T)].
* **Anti-collapse.** Minibatch std-dev in the critic and a mode-seeking term in the generator loss.
* **Learning rates.** Faster critic (η_D = 4e-4, η_G = 1e-4); the proposal's η_G > η_D collapsed.

## Design notes

* The split is grouped by `lesion_id`, so photographs of the same lesion never appear in both
  train and test. The GAN only ever sees the training split.
* The critic has no output activation: for Pearson χ², `dom(f*) = ℝ`, so `T(x, c)` is an unconstrained scalar.
* Every classifier regime gets the same number of optimiser steps, so the comparison measures
  the training data rather than the amount of training.
