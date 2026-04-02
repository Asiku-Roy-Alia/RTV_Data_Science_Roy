"""
model.py — RTV Field Image Classifier
======================================
Architecture: EfficientNet-B0 pretrained on ImageNet, fine-tuned for 9-class
classification of RTV field check-in images.

Design rationale:
  - EfficientNet-B0 chosen over ResNet-50 for its superior accuracy/parameter
    efficiency ratio (82.0% top-1 on ImageNet at 5.3M params vs ResNet-50's
    76.1% at 25.6M). In a deployment context targeting mobile/edge inference
    this matters; lighter models also overfit less on small classes.
  - Two-phase fine-tuning: backbone frozen → train only head for N epochs,
    then unfreeze all layers with discriminative learning rates (lower LR on
    early layers, higher on head). This avoids destroying pretrained features
    early in training while eventually allowing full adaptation.
  - Dropout(0.3) before the final linear layer to regularise the 9-class head,
    particularly important given guinea-pig-shelter has only ~11 training images
    after the 70/15/15 split.
"""

import logging
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

log = logging.getLogger(__name__)

NUM_CLASSES = 9


def build_model(num_classes: int = NUM_CLASSES, dropout: float = 0.3) -> nn.Module:
    """
    Build EfficientNet-B0 with a custom classification head.

    Architecture change from stock EfficientNet-B0:
      Original head: AdaptiveAvgPool2d -> Dropout(0.2) -> Linear(1280, 1000)
      Ours         : AdaptiveAvgPool2d -> Dropout(0.3) -> Linear(1280, num_classes)

    The backbone weights are initialised from IMAGENET1K_V1 (best stable
    checkpoint). The classifier head is randomly initialised and trained from
    scratch.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = models.efficientnet_b0(weights=weights)

    # Replace the classifier head
    in_features = model.classifier[1].in_features   # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    n_params      = sum(p.numel() for p in model.parameters())
    n_trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "EfficientNet-B0 loaded — total params: %s  trainable: %s",
        f"{n_params:,}", f"{n_trainable:,}"
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all layers except the classifier head.
    Used in phase-1 training to warm up the head without disrupting
    ImageNet pretrained features.
    """
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Backbone FROZEN — trainable params: %s", f"{n_trainable:,}")


def unfreeze_all(model: nn.Module) -> None:
    """
    Unfreeze the entire model for phase-2 fine-tuning.
    A lower learning rate is used for the backbone vs the head (set in train.py).
    """
    for param in model.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("All layers UNFROZEN — trainable params: %s", f"{n_trainable:,}")


def get_param_groups(model: nn.Module, head_lr: float, backbone_lr: float) -> list:
    """
    Discriminative learning rates: backbone gets backbone_lr, head gets head_lr.
    This prevents over-writing well-trained early features while allowing full
    fine-tuning of the task-specific head.
    """
    head_params     = list(model.classifier.parameters())
    backbone_params = [p for n, p in model.named_parameters()
                       if "classifier" not in n and p.requires_grad]
    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params,     "lr": head_lr},
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    m = build_model()
    freeze_backbone(m)
    unfreeze_all(m)
    # Verify forward pass
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    assert out.shape == (2, NUM_CLASSES), f"Unexpected output shape: {out.shape}"
    log.info("Forward pass OK — output shape: %s", out.shape)
