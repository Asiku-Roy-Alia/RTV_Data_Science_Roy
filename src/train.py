"""
train.py — Two-Phase Fine-Tuning Training Loop
================================================
Phase 1 (epochs 1–FREEZE_EPOCHS):
    Backbone frozen, only head trained.
    Higher LR (1e-3), fast convergence without destroying pretrained features.

Phase 2 (epochs FREEZE_EPOCHS+1 – MAX_EPOCHS):
    All layers unfrozen, discriminative LRs.
    Backbone LR = 1e-4, head LR = 5e-4.
    CosineAnnealingLR restarts learning rate schedule smoothly.

Early stopping on val weighted-F1 (patience=7) — F1 is the right stopping
metric here because the class imbalance makes accuracy misleading: a model
that ignores guinea-pig-shelter can still hit >97% accuracy.

Checkpointing: best model (by val F1) saved to checkpoints/best_model.pth.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

# ── allow running from task2/ directory or repo root ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "task1"))
from dataset import build_dataloaders          
from model import build_model, freeze_backbone, unfreeze_all, get_param_groups  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── try W&B (optional — training works without it) ───────────────────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    log.warning("wandb not installed — experiment tracking disabled. "
                "Install with: pip install wandb")

# ── hyperparameters ───────────────────────────────────────────────────────────
DEFAULTS = dict(
    data_dir      = "../../data",
    batch_size    = 32,
    freeze_epochs = 5,        # phase 1: head-only training
    max_epochs    = 15,       # total epochs (phase 1 + phase 2)
    head_lr       = 1e-3,     # phase-1 LR (also phase-2 head LR)
    backbone_lr   = 1e-4,     # phase-2 backbone LR (10× lower than head)
    weight_decay  = 1e-4,
    patience      = 7,        # early-stopping patience (epochs without val F1 improvement)
    dropout       = 0.3,
    num_workers   = 4,
    checkpoint_dir= "checkpoints",
    seed          = 42,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np, random
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    if device.type == "cuda":
        log.info("  GPU: %s  VRAM: %.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)
    return device


# ── one epoch of training ─────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, criterion, optimiser, device, epoch, use_wandb
) -> dict:
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    t0 = time.time()

    for step, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        # Gradient clipping — prevents occasional large gradient spikes during
        # phase-2 unfreeze, especially with small minority-class batches.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    n          = len(loader.dataset)
    epoch_loss = running_loss / n
    epoch_f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    elapsed    = time.time() - t0

    log.info(
        "Epoch %3d [TRAIN]  loss=%.4f  weighted_F1=%.4f  time=%.1fs",
        epoch, epoch_loss, epoch_f1, elapsed
    )
    metrics = {"train/loss": epoch_loss, "train/weighted_f1": epoch_f1, "epoch": epoch}
    if use_wandb:
        wandb.log(metrics)
    return metrics


# ── one epoch of validation ───────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, epoch, use_wandb, split="val") -> dict:
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    n          = len(loader.dataset)
    epoch_loss = running_loss / n
    epoch_f1   = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    # Also compute macro F1 — penalises poor performance on minority classes equally.
    macro_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    log.info(
        "Epoch %3d [%s]   loss=%.4f  weighted_F1=%.4f  macro_F1=%.4f",
        epoch, split.upper().ljust(5), epoch_loss, epoch_f1, macro_f1
    )
    metrics = {
        f"{split}/loss"       : epoch_loss,
        f"{split}/weighted_f1": epoch_f1,
        f"{split}/macro_f1"   : macro_f1,
        "epoch"               : epoch,
    }
    if use_wandb:
        wandb.log(metrics)
    return metrics


# ── main training loop ────────────────────────────────────────────────────────

def train(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = get_device()

    # ── W&B initialisation ────────────────────────────────────────────────────
    use_wandb = WANDB_AVAILABLE and os.environ.get("WANDB_DISABLED", "false") != "true"
    if use_wandb:
        wandb.init(
            project = "rtv-image-classifier",
            config  = cfg,
            tags    = ["efficientnet-b0", "field-images", "9-class"],
        )
        log.info("W&B run: %s", wandb.run.url)

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, meta = build_dataloaders(
        data_dir   = cfg["data_dir"],
        batch_size = cfg["batch_size"],
        num_workers= cfg["num_workers"],
    )
    n_classes      = meta["n_classes"]
    weight_tensor  = meta["weight_tensor"].to(device)
    idx_to_class   = meta["idx_to_class"]

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(num_classes=n_classes, dropout=cfg["dropout"]).to(device)

    # ── loss — weighted cross-entropy addresses class imbalance ───────────────
    # weight_tensor is the inverse-frequency vector computed in dataset.py.
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)
    # label_smoothing=0.1: softens hard targets, reduces overconfidence,
    # and has been shown to improve calibration on imbalanced datasets.

    # ── Phase 1 setup: frozen backbone ────────────────────────────────────────
    freeze_backbone(model)
    optimiser = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["head_lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=cfg["freeze_epochs"], eta_min=1e-5)

    # ── checkpoint directory ──────────────────────────────────────────────────
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best_model.pth"

    best_val_f1    = 0.0
    patience_count = 0
    history        = []

    log.info("=" * 60)
    log.info("TRAINING START — %d epochs (%d frozen + %d unfrozen)",
             cfg["max_epochs"], cfg["freeze_epochs"],
             cfg["max_epochs"] - cfg["freeze_epochs"])
    log.info("=" * 60)

    for epoch in range(1, cfg["max_epochs"] + 1):

        # ── Phase transition at freeze_epochs ────────────────────────────────
        if epoch == cfg["freeze_epochs"] + 1:
            log.info("-" * 60)
            log.info("PHASE 2: Unfreezing backbone with discriminative LRs "
                     "(backbone_lr=%.0e  head_lr=%.0e)",
                     cfg["backbone_lr"], cfg["head_lr"])
            log.info("-" * 60)
            unfreeze_all(model)
            param_groups = get_param_groups(model, cfg["head_lr"], cfg["backbone_lr"])
            optimiser    = AdamW(param_groups, weight_decay=cfg["weight_decay"])
            scheduler    = CosineAnnealingLR(
                optimiser,
                T_max = cfg["max_epochs"] - cfg["freeze_epochs"],
                eta_min = 1e-6
            )

        # ── train + validate ─────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimiser, device, epoch, use_wandb
        )
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, use_wandb, split="val"
        )

        scheduler.step()

        val_f1 = val_metrics["val/weighted_f1"]
        history.append({**train_metrics, **val_metrics})

        # ── checkpoint best model ─────────────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_count = 0
            torch.save({
                "epoch"         : epoch,
                "model_state"   : model.state_dict(),
                "optimiser_state": optimiser.state_dict(),
                "val_f1"        : val_f1,
                "cfg"           : cfg,
                "class_to_idx"  : meta["class_to_idx"],
                "idx_to_class"  : idx_to_class,
            }, best_ckpt)
            log.info("  --> New best val F1=%.4f — checkpoint saved", val_f1)
            if use_wandb:
                wandb.run.summary["best_val_f1"]    = val_f1
                wandb.run.summary["best_epoch"]     = epoch
        else:
            patience_count += 1
            log.info("  --> No improvement (%d/%d patience)", patience_count, cfg["patience"])

        # ── early stopping ────────────────────────────────────────────────────
        if patience_count >= cfg["patience"]:
            log.info("Early stopping triggered at epoch %d (best val F1=%.4f)",
                     epoch, best_val_f1)
            break

    # ── final evaluation on test set ─────────────────────────────────────────
    log.info("=" * 60)
    log.info("LOADING BEST CHECKPOINT for test evaluation")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    log.info("Best checkpoint: epoch %d  val_F1=%.4f", ckpt["epoch"], ckpt["val_f1"])

    test_metrics = validate(
        model, test_loader, criterion, device, ckpt["epoch"], use_wandb, split="test"
    )

    # ── save training history ─────────────────────────────────────────────────
    history_path = ckpt_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({"history": history, "test": test_metrics, "cfg": cfg}, f, indent=2)
    log.info("Training history saved to %s", history_path)

    if use_wandb:
        wandb.finish()

    return {
        "best_val_f1"  : best_val_f1,
        "test_metrics" : test_metrics,
        "checkpoint"   : str(best_ckpt),
        "meta"         : meta,
        "model"        : model,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RTV image classifier")
    for k, v in DEFAULTS.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    args   = parser.parse_args()
    cfg    = vars(args)

    log.info("CONFIG: %s", json.dumps(cfg, indent=2))
    results = train(cfg)
    log.info("DONE — best val F1: %.4f", results["best_val_f1"])
    log.info("Test  — weighted F1: %.4f  macro F1: %.4f",
             results["test_metrics"]["test/weighted_f1"],
             results["test_metrics"]["test/macro_f1"])
