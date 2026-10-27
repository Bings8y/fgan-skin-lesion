"""HAM10000 data pipeline: preprocessing cache, lesion-aware split, GPU augmentation."""
import os
import zipfile
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)


def _iter_zip_images(raw_dir):
    for name in sorted(os.listdir(raw_dir)):
        if not name.endswith(".zip"):
            continue
        with zipfile.ZipFile(os.path.join(raw_dir, name)) as zf:
            for member in zf.namelist():
                if member.lower().endswith(".jpg"):
                    image_id = os.path.splitext(os.path.basename(member))[0]
                    with zf.open(member) as fh:
                        yield image_id, Image.open(fh).convert("RGB")


def _square_resize(img, size):
    # Centre-crop to a square (HAM10000 images are 600x450) then resize.
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img.resize((size, size), Image.BICUBIC)


def build_cache(raw_dir, out_path, size=128, test_frac=0.2, seed=42):
    """Decode every image once, resize, and store a uint8 array plus a lesion-grouped split."""
    meta = pd.read_csv(os.path.join(raw_dir, "HAM10000_metadata.csv"))
    meta = meta.drop_duplicates("image_id").set_index("image_id")

    images, ids = [], []
    for image_id, img in _iter_zip_images(raw_dir):
        if image_id in meta.index:
            images.append(np.asarray(_square_resize(img, size), dtype=np.uint8))
            ids.append(image_id)
    images = np.stack(images)
    meta = meta.loc[ids]
    labels = meta["dx"].map(CLASS_TO_IDX).to_numpy(np.int64)

    # Split by lesion_id so that multiple photos of the same lesion never straddle train/test.
    n_splits = int(round(1 / test_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(images, labels, groups=meta["lesion_id"].to_numpy()))
    split = np.zeros(len(ids), dtype=np.int8)  # 0 = train, 1 = test
    split[test_idx] = 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, images=images, labels=labels, split=split, image_ids=np.array(ids))
    return images, labels, split


def load_cache(path, split=None):
    d = np.load(path)
    images, labels, s = d["images"], d["labels"], d["split"]
    if split == "train":
        m = s == 0
    elif split == "test":
        m = s == 1
    else:
        m = np.ones_like(s, dtype=bool)
    return images[m], labels[m]


def to_tensor(images_uint8):
    """NHWC uint8 -> NCHW float in [-1, 1]."""
    x = torch.from_numpy(images_uint8).permute(0, 3, 1, 2).float()
    return x / 127.5 - 1.0


def class_counts(labels):
    return np.bincount(labels, minlength=NUM_CLASSES)


def tempered_sampling_weights(labels, power=0.5):
    """Per-sample weights so class c is drawn with probability proportional to n_c ** power."""
    counts = class_counts(labels).astype(np.float64)
    class_p = counts ** power
    class_p /= class_p.sum()
    return class_p[labels] / counts[labels]


@torch.no_grad()
def gpu_augment(x, max_rot_deg=20.0, jitter=0.05):
    """Dermoscopy-safe augmentation on a [-1, 1] batch: flips, rotation, mild colour jitter.

    Lesions have no canonical orientation, so axial flips and small rotations preserve
    diagnostic morphology; colour jitter is kept small so pigment cues stay intact.
    """
    n = x.size(0)
    dev = x.device
    flip_h = torch.rand(n, device=dev) < 0.5
    flip_v = torch.rand(n, device=dev) < 0.5
    x = torch.where(flip_h[:, None, None, None], x.flip(3), x)
    x = torch.where(flip_v[:, None, None, None], x.flip(2), x)

    theta = (torch.rand(n, device=dev) * 2 - 1) * math.radians(max_rot_deg)
    cos, sin = torch.cos(theta), torch.sin(theta)
    zeros = torch.zeros_like(cos)
    mat = torch.stack([torch.stack([cos, -sin, zeros], 1), torch.stack([sin, cos, zeros], 1)], 1)
    grid = F.affine_grid(mat, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    if jitter > 0:
        bright = (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * jitter * 2
        contrast = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * jitter
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * contrast + mean + bright
    return x.clamp(-1, 1)
