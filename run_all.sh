#!/usr/bin/env bash
# End-to-end pipeline: data -> f-GAN -> synthetic balancing -> downstream classifiers -> figures.
set -euo pipefail
cd "$(dirname "$0")"

ITERS=${ITERS:-8000}
SEEDS=${SEEDS:-"0 1 2"}

# 1. Data (download HAM10000 from Harvard Dataverse doi:10.7910/DVN/DBW86T into data/raw first)
[ -f data/ham10000_128.npz ] || python prepare_data.py

# 2. f-GAN training: Pearson chi^2 (main) and Jensen-Shannon (ablation, shorter).
#    Shared stabilisers: non-saturating G loss, TTUR (faster critic), spectral norm, minibatch std-dev,
#    mode-seeking term. For Pearson, --drift 0.25 turns the critic into a chi^2-to-the-mixture estimator.
COMMON="--g_loss nonsat --lr_d 4e-4 --lr_g 1e-4 --pool sum --extra_from 2 --mbstd 4 --ms 1.0"
python train_fgan.py --div pearson --drift 0.25 --iters "$ITERS" --snapshots 3000 6000 $COMMON --out runs/fgan_pearson
python train_fgan.py --div js --iters 3000 $COMMON --out runs/fgan_js

# 3. Synthetic data: balancing set for the classifier, fixed per-class set for figures
python generate_synthetic.py --mode balance --target 2000 --out runs/fgan_pearson/synthetic_balance.npz
python generate_synthetic.py --mode balance --target 1000 --out runs/fgan_pearson/synthetic_balance_1000.npz
python generate_synthetic.py --mode fixed --per_class 200 --out runs/fgan_pearson/synthetic_fixed.npz

# 4. Sample quality (final Pearson model, plus matched-iteration Pearson vs JS comparison)
python evaluate_gan.py --reference --ckpt pearson=runs/fgan_pearson/checkpoint.pt \
    pearson_3k=runs/fgan_pearson/generator_ema_003000.pt js_3k=runs/fgan_js/checkpoint.pt

# 5. Downstream ResNet-18 benchmarking on the untouched real test split
for seed in $SEEDS; do
  for regime in baseline ros fgan; do
    python train_classifier.py --regime "$regime" --seed "$seed"
  done
  python train_classifier.py --regime fgan --synthetic runs/fgan_pearson/synthetic_balance_1000.npz \
      --name fgan1000 --seed "$seed"
done

# 6. Figures and summary tables
python make_figures.py

# 7. Report: build, render with LibreOffice, read page numbers back for the table of contents, rebuild
python report/build_report.py
soffice --headless --convert-to pdf FGAN_Skin_Lesion_E-Report.docx
python report/paginate.py FGAN_Skin_Lesion_E-Report.pdf
python report/build_report.py --pages report/pages.json
soffice --headless --convert-to pdf FGAN_Skin_Lesion_E-Report.docx

# 8. Conference paper (two-column IEEE-style .docx, rendered to PDF)
python journal/build_conference_docx.py
(cd journal && soffice --headless --convert-to pdf FGAN_Conference_Paper.docx)
