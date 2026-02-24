import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
    reduction: str = "none",
) -> torch.Tensor:
    """Focal loss for binary classification with logits input."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = targets * p + (1.0 - targets) * (1.0 - p)
    focal_weight = (1.0 - pt) ** gamma
    alpha_weight = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    loss = focal_weight * alpha_weight * bce

    if reduction == "mean":
        return loss.mean()
    return loss


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Dice loss for binary mask prediction."""
    probs = torch.sigmoid(logits)
    probs_flat = probs.reshape(-1)
    targets_flat = targets.reshape(-1)
    intersection = (probs_flat * targets_flat).sum()
    union = probs_flat.sum() + targets_flat.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def consistency_reg(
    binary_logits: torch.Tensor,
    img_ground_logits: torch.Tensor,
    txt_ground_logits: torch.Tensor,
    has_img: torch.Tensor,
    has_txt: torch.Tensor,
    text_grounding_labels: torch.Tensor = None,
) -> torch.Tensor:
    """Encourage grounding signals to align with detection confidence for mixed samples."""
    mixed = has_img & has_txt
    if not mixed.any():
        return torch.tensor(0.0, device=binary_logits.device, requires_grad=False)

    det_conf = torch.sigmoid(binary_logits[mixed, 1])
    img_max_prob = torch.sigmoid(img_ground_logits[mixed]).max(dim=-1).values

    # For text: mask out special/padding tokens (labeled -100) to avoid
    # spurious peaks from pad positions
    txt_logits_mixed = txt_ground_logits[mixed]  # (N_mixed, 128)
    if text_grounding_labels is not None:
        txt_labels_mixed = text_grounding_labels[mixed]  # (N_mixed, 128)
        valid_mask = (txt_labels_mixed != -100.0)  # True for content tokens
        # Replace invalid positions with large negative so sigmoid -> ~0
        txt_logits_masked = txt_logits_mixed.masked_fill(~valid_mask, -1e9)
    else:
        txt_logits_masked = txt_logits_mixed

    txt_max_prob = torch.sigmoid(txt_logits_masked).max(dim=-1).values

    loss_img = (det_conf * (1.0 - img_max_prob)).mean()
    loss_txt = (det_conf * (1.0 - txt_max_prob)).mean()

    return loss_img + loss_txt


class MultiTaskLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.lambda_binary = config.get("lambda_binary", 1.0)
        self.lambda_type = config.get("lambda_type", 0.3)
        self.lambda_text_grounding = config.get("lambda_text_grounding", 0.5)
        self.lambda_image_grounding = config.get("lambda_image_grounding", 0.5)
        self.lambda_consistency = config.get("lambda_consistency", 0.1)

        self.focal_gamma = config.get("focal_gamma", 2.0)
        self.focal_alpha = config.get("focal_alpha", 0.75)

        self.dice_weight = config.get("dice_weight", 0.3)
        self.bce_weight = 1.0 - self.dice_weight

        self.type_class_weights = None

    def set_type_class_weights(self, weights: torch.Tensor):
        self.register_buffer("_type_weights", weights)
        self.type_class_weights = weights

    def forward(self, outputs: dict, batch: dict) -> dict:
        device = outputs["binary_logits"].device

        # Binary detection loss
        loss_binary = F.cross_entropy(
            outputs["binary_logits"], batch["binary_labels"]
        )

        # Type classification loss
        type_weights = self.type_class_weights
        if type_weights is not None:
            type_weights = type_weights.to(device)
        loss_type = F.cross_entropy(
            outputs["type_logits"],
            batch["type_labels"],
            weight=type_weights,
        )

        # Text grounding loss (focal BCE, masked)
        loss_tg = torch.tensor(0.0, device=device)
        has_tg = batch["has_text_grounding"]
        if has_tg.any():
            tg_logits = outputs["text_ground_logits"][has_tg]
            tg_labels = batch["text_grounding_labels"][has_tg]

            valid = (tg_labels != -100.0)
            raw = focal_bce_with_logits(
                tg_logits, tg_labels.clamp(min=0.0),
                gamma=self.focal_gamma, alpha=self.focal_alpha,
            )
            loss_tg = (raw * valid.float()).sum() / valid.float().sum().clamp(min=1.0)

        # Image grounding loss (BCE + Dice, masked)
        loss_ig = torch.tensor(0.0, device=device)
        has_ig = batch["has_image_grounding"]
        if has_ig.any():
            ig_logits = outputs["image_ground_logits"][has_ig]
            ig_labels = batch["image_grounding_labels"][has_ig]

            bce_ig = F.binary_cross_entropy_with_logits(ig_logits, ig_labels)
            dice_ig = dice_loss(ig_logits, ig_labels)
            loss_ig = self.bce_weight * bce_ig + self.dice_weight * dice_ig

        # Consistency regularizer
        loss_consist = consistency_reg(
            outputs["binary_logits"],
            outputs["image_ground_logits"],
            outputs["text_ground_logits"],
            has_ig, has_tg,
            text_grounding_labels=batch["text_grounding_labels"],
        )

        total = (
            self.lambda_binary * loss_binary
            + self.lambda_type * loss_type
            + self.lambda_text_grounding * loss_tg
            + self.lambda_image_grounding * loss_ig
            + self.lambda_consistency * loss_consist
        )

        return {
            "total": total,
            "binary": loss_binary.detach(),
            "type": loss_type.detach(),
            "text_grounding": loss_tg.detach() if isinstance(loss_tg, torch.Tensor) else torch.tensor(0.0),
            "image_grounding": loss_ig.detach() if isinstance(loss_ig, torch.Tensor) else torch.tensor(0.0),
            "consistency": loss_consist.detach(),
        }
