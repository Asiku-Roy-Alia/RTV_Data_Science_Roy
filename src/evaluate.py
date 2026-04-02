"""
evaluate.py — Evaluation, Confusion Matrix & Report
=====================================================
Loads the best checkpoint, runs inference on the test set, and produces:
  1. Per-class precision, recall, F1 (printed + saved as JSON)
  2. confusion_matrix.png (row-normalised) and confusion_matrix_raw.png
  3. Top-N misclassifications logged per class
  4. evaluation_report.json — machine-readable results for write-up

Run:
    python evaluate.py --checkpoint checkpoints/best_model.pth --data_dir ../data
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from collections import Counter

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "task1"))
from dataset import build_dataloaders
from model import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    """Return (all_preds, all_labels, all_probs) arrays over the full loader."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1).cpu()
        preds  = logits.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.tolist())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ── confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    output_path: str = "checkpoints/confusion_matrix.png",
    normalise: bool = True,
) -> None:
    """
    Save a confusion matrix heatmap.

    Row-normalised version shows recall per class, which is more interpretable
    than raw counts when classes are heavily imbalanced (guinea-pig-shelter has
    only 2 test samples vs 23 for majority classes).

    Safe divide: np.divide with 'where' emits a UserWarning due to
    uninitialised memory in masked positions. We use zeros_like +
    masked assignment instead, which is unambiguous and warning-free.
    """
    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot  = np.zeros_like(cm, dtype=float)
        nonzero  = row_sums.flatten() != 0
        cm_plot[nonzero] = cm[nonzero] / row_sums[nonzero]
        fmt   = ".2f"
        title = "Confusion Matrix (row-normalised — recall per class)"
    else:
        cm_plot = cm.astype(float)
        fmt     = ".0f"
        title   = "Confusion Matrix (raw counts)"

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm_plot,
        annot       = True,
        fmt         = fmt,
        cmap        = "Blues",
        xticklabels = class_names,
        yticklabels = class_names,
        linewidths  = 0.4,
        linecolor   = "lightgrey",
        ax          = ax,
        vmin        = 0,
        vmax        = 1 if normalise else None,
    )
    ax.set_xlabel("Predicted class", fontsize=12, labelpad=10)
    ax.set_ylabel("True class",      fontsize=12, labelpad=10)
    ax.set_title(title,              fontsize=13, pad=14)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Confusion matrix saved to %s", output_path)


# ── misclassification analysis ────────────────────────────────────────────────

def analyse_errors(
    preds: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    idx_to_class: dict,
    top_n: int = 3,
) -> dict:
    """
    For each class, find the top-N most common misclassification targets.
    Returns dict: class_name -> list of (confused_with_name, count).
    """
    errors = {}
    for true_idx in range(len(idx_to_class)):
        cls_name   = idx_to_class[true_idx]
        mask       = labels == true_idx
        if mask.sum() == 0:
            errors[cls_name] = []
            continue
        wrong_mask = mask & (preds != labels)
        if wrong_mask.sum() == 0:
            errors[cls_name] = []
            continue
        counts = Counter(preds[wrong_mask].tolist())
        errors[cls_name] = [
            (idx_to_class[k], v) for k, v in counts.most_common(top_n)
        ]
    return errors


# ── main ──────────────────────────────────────────────────────────────────────

def evaluate(checkpoint_path: str, data_dir: str, output_dir: str = "checkpoints") -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Evaluating on device: %s", device)

    # load checkpoint
    ckpt         = torch.load(checkpoint_path, map_location=device)
    cfg          = ckpt.get("cfg", {})
    idx_to_class = ckpt["idx_to_class"]
    class_names  = [idx_to_class[i] for i in range(len(idx_to_class))]
    log.info("Checkpoint: epoch=%d  val_F1=%.4f", ckpt["epoch"], ckpt["val_f1"])

    # build model and load weights
    model = build_model(num_classes=len(class_names), dropout=cfg.get("dropout", 0.3))
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    # data
    _, _, test_loader, meta = build_dataloaders(
        data_dir=data_dir, batch_size=64, num_workers=4,
    )

    # inference
    log.info("Running inference on test set (%d samples)...", len(meta["test_samples"]))
    preds, labels, probs = run_inference(model, test_loader, device)

    # scalar metrics
    acc      = accuracy_score(labels, preds)
    w_f1     = f1_score(labels, preds, average="weighted",  zero_division=0)
    macro_f1 = f1_score(labels, preds, average="macro",     zero_division=0)
    w_prec   = precision_score(labels, preds, average="weighted", zero_division=0)
    w_rec    = recall_score(labels, preds,    average="weighted", zero_division=0)

    log.info("=" * 60)
    log.info("TEST SET RESULTS")
    log.info("  Accuracy          : %.4f", acc)
    log.info("  Weighted F1       : %.4f  (primary metric)", w_f1)
    log.info("  Macro F1          : %.4f  (penalises minority failures equally)", macro_f1)
    log.info("  Weighted Precision: %.4f", w_prec)
    log.info("  Weighted Recall   : %.4f", w_rec)
    log.info("=" * 60)

    # per-class report
    report = classification_report(
        labels, preds, target_names=class_names, zero_division=0, digits=4,
    )
    log.info("PER-CLASS REPORT:\n%s", report)

    # confusion matrices
    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, class_names,
        output_path=str(Path(output_dir) / "confusion_matrix.png"),
        normalise=True)
    plot_confusion_matrix(cm, class_names,
        output_path=str(Path(output_dir) / "confusion_matrix_raw.png"),
        normalise=False)

    # misclassification patterns
    error_analysis = analyse_errors(preds, labels, probs, idx_to_class)
    log.info("TOP MISCLASSIFICATION PATTERNS (true class -> most confused with):")
    for cls, confused in error_analysis.items():
        if confused:
            pairs = ", ".join(f"{c} ({n})" for c, n in confused)
            log.info("  %-25s -> %s", cls, pairs)
        else:
            log.info("  %-25s -> no errors", cls)

    # per-class dict
    per_class_metrics = {}
    for i, cls in enumerate(class_names):
        mask = labels == i
        if mask.sum() == 0:
            per_class_metrics[cls] = {"support": 0, "f1": None,
                                       "precision": None, "recall": None}
            continue
        per_class_metrics[cls] = {
            "support"  : int(mask.sum()),
            "f1"       : round(float(f1_score(labels==i, preds==i, zero_division=0)), 4),
            "precision": round(float(precision_score(labels==i, preds==i, zero_division=0)), 4),
            "recall"   : round(float(recall_score(labels==i, preds==i, zero_division=0)), 4),
        }

    # save report
    report_data = {
        "checkpoint"       : checkpoint_path,
        "best_epoch"       : ckpt["epoch"],
        "val_f1_at_ckpt"   : round(ckpt["val_f1"], 4),
        "test": {
            "accuracy"          : round(acc,      4),
            "weighted_f1"       : round(w_f1,     4),
            "macro_f1"          : round(macro_f1, 4),
            "weighted_precision": round(w_prec,   4),
            "weighted_recall"   : round(w_rec,    4),
        },
        "per_class"        : per_class_metrics,
        "confusion_matrix" : cm.tolist(),
        "class_names"      : class_names,
        "top_misclassifications": {
            k: [(c, n) for c, n in v] for k, v in error_analysis.items()
        },
    }
    report_path = Path(output_dir) / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
    log.info("Full evaluation report saved to %s", report_path)

    return report_data


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RTV image classifier")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--data_dir",   default="../../data")
    parser.add_argument("--output_dir", default="checkpoints")
    args = parser.parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )