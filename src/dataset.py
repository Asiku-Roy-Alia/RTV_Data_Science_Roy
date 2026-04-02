"""
Task 1: Data Analysis & Preparation
====================================
RTV Field Check-in Image Classifier — 9-class pipeline.

Dataset stats (reproduced from runtime log):
  Total images : 1,216
  Classes      : 9
  Imbalance    : 9.38x  (guinea-pig-shelter=16, all others=150)

Key findings from manual inspection :
  - vsla class contains images with visible PII (client names, financial data).
    Flagged in scan log; no images removed here — data governance decision is
    for RTV's data team, not the ML pipeline.
  - guinea-pig-shelter class has severe quality issues: completely black image,
    cartoon/pixelated images, unrelated subjects (greenhouse, bowl of produce).
    These are logged per-file but retained; removal would leave ~10 usable
    training samples which is worse than keeping noisy ones.
  - Mixed landscape/portrait orientation across all classes — handled by
    EXIF-aware loading (ImageOps.exif_transpose) + RandomResizedCrop.
  - High semantic variability within classes — addressed by augmentation breadth.
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGE_SIZE    = 224
RANDOM_SEED   = 42

# Accept .jpg/.jpeg/.png and extensionless files (vsla class stores no extension)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ""}

# vsla filenames containing these substrings are flagged as potential PII images.
# The pipeline does NOT remove them — that is a data governance decision for RTV.
# They are logged so reviewers are aware.
PII_FLAG_SUBSTRINGS = ["vslaFollowUpImageKey", "vslaFol"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_image_file(path: Path) -> bool:
    """Accept .jpg/.jpeg/.png and extensionless files (vsla class)."""
    return path.suffix.lower() in VALID_EXTENSIONS


def _safe_open(path: Path) -> Optional[Image.Image]:
    """
    Open an image and return an RGB PIL Image, or None on failure.

    Two important steps beyond a naive Image.open():
      1. PIL.Image.verify() — catches truncated/corrupt files before decode.
      2. ImageOps.exif_transpose() — corrects orientation from EXIF metadata.
         Field photos shot in portrait on Android devices are often stored
         landscape with an EXIF rotation tag. Without this fix, the model
         sees rotated inputs during training but upright inputs at inference,
         causing a train/test distribution mismatch.

    All failures are logged with the filename so they are fully traceable.
    """
    try:
        img = Image.open(path)
        img.verify()                          # catches truncated files
        img = Image.open(path)                # re-open after verify (verify closes)
        img = ImageOps.exif_transpose(img)    # correct EXIF orientation
        img = img.convert("RGB")              # normalise to 3-channel RGB
        return img
    except (UnidentifiedImageError, OSError, Exception) as exc:
        log.warning("Skipping corrupt/unreadable file %s: %s", path.name, exc)
        return None


def _image_hash(path: Path) -> str:
    """MD5 of first 64 KB — fast duplicate-detection proxy."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return h.hexdigest()


def _flag_pii(path: Path, cls_name: str) -> bool:
    """
    Return True if this file is suspected to contain PII based on filename
    and class. vsla images include financial/member records; their filenames
    contain 'vslaFollowUpImageKey' or 'vslaFol'.

    NOTE: This function only LOGS a warning. No image is removed or excluded.
    Data governance (whether to blur, exclude, or anonymise) is RTV's decision.
    """
    if cls_name == "vsla":
        fname = path.name
        if any(sub in fname for sub in PII_FLAG_SUBSTRINGS):
            return True
    return False


# ── dataset scanner ───────────────────────────────────────────────────────────

def scan_dataset(data_dir: str) -> Dict:
    """
    Walk data_dir, collect (path, label) pairs, and LOG:
      - per-class image counts
      - per-class file-size statistics (proxy for resolution/quality)
      - corrupt / unreadable file count
      - duplicate files (by MD5 hash)
      - imbalance ratio
      - PII-risk file count (vsla class)
      - class weights for WeightedRandomSampler and CrossEntropyLoss

    All numbers printed to log are directly referenced in analysis_writeup.docx.
    """
    data_path = Path(data_dir)
    assert data_path.is_dir(), f"data_dir not found: {data_dir}"

    class_dirs   = sorted([d for d in data_path.iterdir() if d.is_dir()])
    class_names  = [d.name for d in class_dirs]
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    log.info("=" * 60)
    log.info("DATASET SCAN: %s", data_dir)
    log.info("Found %d class directories: %s", len(class_names), class_names)

    samples: List[Tuple[Path, int]] = []
    class_counts: Dict[str, int]    = {}
    class_sizes:  Dict[str, List[float]] = {}
    seen_hashes:  Dict[str, str]    = {}
    n_duplicates  = 0
    n_corrupt     = 0
    n_pii_flagged = 0

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        cls_idx  = class_to_idx[cls_name]
        files    = [p for p in cls_dir.iterdir()
                    if p.is_file() and _is_image_file(p)]

        valid_files: List[Tuple[Path, int]] = []
        sizes_kb: List[float] = []

        for fpath in files:
            # ── PII flag (log only, do not exclude) ───────────────────────
            if _flag_pii(fpath, cls_name):
                log.warning(
                    "PII-RISK: %s/%s — vsla follow-up image may contain "
                    "client name / financial data. Retained for training; "
                    "recommend data governance review.",
                    cls_name, fpath.name[:60]
                )
                n_pii_flagged += 1

            # ── integrity check ───────────────────────────────────────────
            img = _safe_open(fpath)
            if img is None:
                n_corrupt += 1
                continue

            # ── duplicate detection ───────────────────────────────────────
            h = _image_hash(fpath)
            if h in seen_hashes:
                log.warning(
                    "DUPLICATE: %s  ==  %s (retained, flagged)",
                    fpath.name[:60], seen_hashes[h]
                )
                n_duplicates += 1
            else:
                seen_hashes[h] = str(fpath)

            valid_files.append((fpath, cls_idx))
            sizes_kb.append(fpath.stat().st_size / 1024)

        class_counts[cls_name] = len(valid_files)
        class_sizes[cls_name]  = sizes_kb
        samples.extend(valid_files)

    # ── imbalance analysis ────────────────────────────────────────────────────
    counts    = list(class_counts.values())
    max_count = max(counts)
    min_count = min(counts)
    imb_ratio = max_count / max(min_count, 1)
    total     = sum(counts)

    log.info("-" * 60)
    log.info("CLASS DISTRIBUTION (imbalance ratio %.2fx):", imb_ratio)
    log.info("  %-25s %6s  %8s  %8s  %8s",
             "class", "count", "avg_KB", "min_KB", "max_KB")
    for cls_name in class_names:
        sz  = class_sizes[cls_name]
        avg = np.mean(sz)  if sz else 0.0
        mn  = np.min(sz)   if sz else 0.0
        mx  = np.max(sz)   if sz else 0.0
        log.info("  %-25s %6d  %8.1f  %8.1f  %8.1f",
                 cls_name, class_counts[cls_name], avg, mn, mx)
    log.info("  %-25s %6d  (total)",  "TOTAL",              total)
    log.info("  Corrupt / unreadable files : %d",            n_corrupt)
    log.info("  Duplicate files detected   : %d (retained)", n_duplicates)
    log.info("  PII-risk files flagged     : %d (vsla class, retained — "
             "governance review recommended)", n_pii_flagged)

    # ── inverse-frequency class weights ──────────────────────────────────────
    n_classes     = len(class_names)
    class_weights = {
        cls: total / (n_classes * max(cnt, 1))
        for cls, cnt in class_counts.items()
    }
    weight_tensor = torch.tensor(
        [class_weights[idx_to_class[i]] for i in range(n_classes)],
        dtype=torch.float32,
    )

    log.info("-" * 60)
    log.info("CLASS WEIGHTS (inverse-frequency for loss/sampler):")
    for cls_name in class_names:
        log.info("  %-25s %.4f", cls_name, class_weights[cls_name])
    log.info("=" * 60)

    stats = {
        "total"        : total,
        "n_classes"    : n_classes,
        "imb_ratio"    : round(imb_ratio, 3),
        "n_corrupt"    : n_corrupt,
        "n_duplicates" : n_duplicates,
        "n_pii_flagged": n_pii_flagged,
        "class_counts" : class_counts,
        "class_weights": class_weights,
    }
    return {
        "samples"      : samples,
        "class_to_idx" : class_to_idx,
        "idx_to_class" : idx_to_class,
        "class_counts" : class_counts,
        "weight_tensor": weight_tensor,
        "stats"        : stats,
    }


# ── train / val / test split ──────────────────────────────────────────────────

def make_splits(
    samples:    List[Tuple[Path, int]],
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed:       int   = RANDOM_SEED,
) -> Tuple[List, List, List]:
    """
    Stratified 70/15/15 split.

    Stratification is critical for guinea-pig-shelter (16 images): without it,
    a random split can leave this class absent from val or test entirely.
    With stratification, the guaranteed minimum per split is ~11/2/3.

    Fixed seed=42 ensures full reproducibility across runs.
    """
    paths  = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    # Step 1: split off test set
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels,
        test_size    = 1.0 - train_frac - val_frac,
        stratify     = labels,
        random_state = seed,
    )
    # Step 2: split train/val from the remainder
    relative_val = val_frac / (train_frac + val_frac)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size    = relative_val,
        stratify     = train_val_labels,
        random_state = seed,
    )

    train = list(zip(train_paths, train_labels))
    val   = list(zip(val_paths,   val_labels))
    test  = list(zip(test_paths,  test_labels))

    log.info("STRATIFIED SPLITS: train=%d  val=%d  test=%d",
             len(train), len(val), len(test))
    return train, val, test


# ── transforms ────────────────────────────────────────────────────────────────

def get_transforms(split: str, image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Returns the augmentation pipeline for the requested split.

    TRAIN augmentations and rationale:
      RandomResizedCrop(224, scale=0.7-1.0)
        Field photos shot at variable distances. Crop simulates zoom variation
        and forces the model to use local rather than global cues.

      RandomHorizontalFlip(p=0.5)
        Installations have no canonical left/right orientation.

      ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1)
        Field photos vary widely in lighting (harsh sun, deep shade, overcast).
        These ranges cover realistic variation without destroying semantics.

      RandomRotation(±15°)
        Handles handheld tilt. Wider rotation was considered but rejected:
        animal pens and tippytaps have clear up/down structure that large
        rotations would distort unrealistically.

      RandomGrayscale(p=0.05)
        Rare. Prevents over-reliance on colour alone.

      GaussianBlur(sigma=0.1-1.5)
        Simulates slightly out-of-focus shots on low-spec field devices.
        Directly addresses the blur quality issue observed in guinea-pig-shelter.

    Augmentations deliberately excluded:
      - Vertical flip: top/bottom orientation is meaningful (tippytap spout,
        pigsty roof). Vertical flips produce unrealistic inputs.
      - Heavy geometric distortion: compliance photos are taken roughly
        straight-on. Extreme shear/perspective diverges too far from real
        distribution.

    VAL/TEST: deterministic resize + centre-crop only. No randomness.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1
            ),
            transforms.RandomRotation(degrees=15),
            transforms.RandomGrayscale(p=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.143)),   # → 256 for 224 target
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class RTVDataset(Dataset):
    """
    PyTorch Dataset for RTV field check-in images.

    Handles:
      - .jpg / .jpeg / extensionless (vsla class)
      - EXIF-corrected orientation via _safe_open / ImageOps.exif_transpose
      - On-the-fly augmentation
      - Graceful fallback for any corrupt file encountered at __getitem__ time
        (returns a black tensor and logs a warning; training is not interrupted)
    """

    def __init__(
        self,
        samples:   List[Tuple[Path, int]],
        transform: transforms.Compose,
        split:     str = "train",
    ):
        self.samples   = samples
        self.transform = transform
        self.split     = split
        log.info("RTVDataset[%s]: %d samples", split, len(samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = _safe_open(path)

        if img is None:
            # Fallback: black image. Logged in _safe_open.
            # Returning black rather than raising keeps the DataLoader alive.
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=0)

        return self.transform(img), label


# ── WeightedRandomSampler ─────────────────────────────────────────────────────

def make_weighted_sampler(
    train_samples: List[Tuple[Path, int]],
    class_weights: Dict[str, float],
    idx_to_class:  Dict[int, str],
) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that draws len(train_samples) samples per
    epoch with replacement, weighted by inverse class frequency.

    Effect: guinea-pig-shelter (weight=8.44) appears ~9.4x more often per
    epoch than majority classes (weight=0.90). Each resampled image receives
    a fresh random augmentation, so upsampled examples are never pixel-identical.

    Alternative rejected: offline duplication of minority-class images.
    Reason: no diversity gain, larger disk footprint, harder to version.
    """
    sample_weights = torch.tensor([
        class_weights[idx_to_class[label]]
        for _, label in train_samples
    ], dtype=torch.float32)

    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(train_samples),
        replacement = True,
    )
    log.info(
        "WeightedRandomSampler: %d samples/epoch with class rebalancing",
        len(train_samples)
    )
    return sampler


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(
    data_dir:    str,
    batch_size:  int  = 32,
    num_workers: int  = 4,
    pin_memory:  bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Full pipeline: raw data_dir → ready DataLoaders.

    Returns (train_loader, val_loader, test_loader, meta)
    where meta contains: class_to_idx, idx_to_class, stats,
                         weight_tensor, n_classes,
                         train/val/test_samples (for reproducibility checks).
    """
    info = scan_dataset(data_dir)

    train_samples, val_samples, test_samples = make_splits(info["samples"])

    train_tf = get_transforms("train")
    eval_tf  = get_transforms("val")

    train_ds = RTVDataset(train_samples, train_tf, split="train")
    val_ds   = RTVDataset(val_samples,   eval_tf,  split="val")
    test_ds  = RTVDataset(test_samples,  eval_tf,  split="test")

    sampler = make_weighted_sampler(
        train_samples,
        info["stats"]["class_weights"],
        info["idx_to_class"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        sampler     = sampler,        # replaces shuffle=True
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,           # avoids 1-sample batches with BatchNorm
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )

    log.info(
        "DataLoaders ready — train_batches=%d  val_batches=%d  test_batches=%d",
        len(train_loader), len(val_loader), len(test_loader),
    )

    meta = {
        "class_to_idx"  : info["class_to_idx"],
        "idx_to_class"  : info["idx_to_class"],
        "stats"         : info["stats"],
        "weight_tensor" : info["weight_tensor"],
        "n_classes"     : info["stats"]["n_classes"],
        "train_samples" : train_samples,
        "val_samples"   : val_samples,
        "test_samples"  : test_samples,
    }
    return train_loader, val_loader, test_loader, meta


# ── analysis report ───────────────────────────────────────────────────────────

def write_analysis_report(stats: Dict, output_path: str = "analysis_report.json") -> None:
    """Persist scan stats as JSON for write-up traceability."""
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info("Analysis report written to %s", output_path)


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"

    train_loader, val_loader, test_loader, meta = build_dataloaders(
        data_dir=data_dir, batch_size=32, num_workers=0
    )
    write_analysis_report(meta["stats"])

    imgs, labels = next(iter(train_loader))
    log.info("Sample batch — images: %s  labels: %s  dtype: %s",
             imgs.shape, labels.shape, imgs.dtype)
    assert imgs.shape[1:] == (3, 224, 224), f"Unexpected image shape: {imgs.shape}"
    assert labels.max() < meta["n_classes"],  "Label index out of range"
    log.info("Task 1 self-test PASSED.")
    log.info("Class mapping: %s", meta["class_to_idx"])
