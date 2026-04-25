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


def bbox_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert raw cxcywh logits to valid normalized xyxy boxes."""
    coords = torch.sigmoid(logits)
    cx, cy, w, h = coords.unbind(dim=-1)
    half_w = 0.5 * w
    half_h = 0.5 * h
    x1 = (cx - half_w).clamp(0.0, 1.0)
    y1 = (cy - half_h).clamp(0.0, 1.0)
    x2 = (cx + half_w).clamp(0.0, 1.0)
    y2 = (cy + half_h).clamp(0.0, 1.0)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0.0)
    return wh[:, 0] * wh[:, 1]


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise generalized IoU for normalized xyxy boxes."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)

    enc_lt = torch.minimum(boxes1[:, None, :2], boxes2[:, :2])
    enc_rb = torch.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0.0)
    enc_area = enc_wh[:, :, 0] * enc_wh[:, :, 1]

    return iou - (enc_area - union) / enc_area.clamp(min=1e-6)


class MultiTaskLoss(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.lambda_binary = config.get("lambda_binary", 1.0)
        self.lambda_type = config.get("lambda_type", 0.3)
        self.lambda_text_grounding = config.get("lambda_text_grounding", 0.5)
        self.lambda_image_grounding = config.get("lambda_image_grounding", 0.5)
        self.lambda_alignment = config.get("lambda_alignment", 0.05)
        self.lambda_itm = config.get("lambda_itm", 0.2)
        self.lambda_consistency = config.get("lambda_consistency", 0.0)

        self.focal_gamma = config.get("focal_gamma", 2.0)
        self.focal_alpha = config.get("focal_alpha", 0.75)
        self.contrastive_temp = config.get("contrastive_temp", 0.07)

        self.dice_weight = config.get("dice_weight", 0.3)
        self.bce_weight = 1.0 - self.dice_weight
        self.bbox_l1_weight = config.get("bbox_l1_weight", 5.0)
        self.bbox_giou_weight = config.get("bbox_giou_weight", 2.0)

        self.type_pos_weights = None

    def set_type_pos_weights(self, weights: torch.Tensor):
        if "_type_pos_weights" in self._buffers:
            self._buffers["_type_pos_weights"] = weights
        else:
            self.register_buffer("_type_pos_weights", weights)
        self.type_pos_weights = weights

    def set_type_class_weights(self, weights: torch.Tensor):
        # Backward-compatible alias for older training/eval scripts.
        self.set_type_pos_weights(weights)

    def forward(self, outputs: dict, batch: dict) -> dict:
        device = outputs["binary_logits"].device

        # Binary detection loss
        loss_binary = F.cross_entropy(
            outputs["binary_logits"], batch["binary_labels"]
        )

        # 4-label manipulation type loss
        type_pos_weights = self.type_pos_weights
        if type_pos_weights is not None:
            type_pos_weights = type_pos_weights.to(device)
        loss_type = F.binary_cross_entropy_with_logits(
            outputs["type_logits"],
            batch["type_labels"],
            pos_weight=type_pos_weights,
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

        # Image grounding loss (bbox L1 + GIoU, masked)
        loss_ig = torch.tensor(0.0, device=device)
        has_ig = batch["has_image_grounding"]
        if has_ig.any():
            pred_boxes = bbox_from_logits(outputs["image_ground_logits"][has_ig])
            target_boxes = batch["image_grounding_labels"][has_ig]

            l1_ig = F.l1_loss(pred_boxes, target_boxes)
            giou = generalized_box_iou(pred_boxes, target_boxes).diagonal()
            giou_ig = (1.0 - giou).mean()
            loss_ig = self.bbox_l1_weight * l1_ig + self.bbox_giou_weight * giou_ig

        # Image-text alignment loss over pristine/original pairs only
        loss_alignment = torch.tensor(0.0, device=device)
        align_logits_t2i = outputs.get("alignment_logits_t2i")
        align_logits_i2t = outputs.get("alignment_logits_i2t")
        align_targets = outputs.get("alignment_targets")
        if (
            align_logits_t2i is not None
            and align_logits_i2t is not None
            and align_targets is not None
        ):
            logits_t2i = align_logits_t2i / self.contrastive_temp
            logits_i2t = align_logits_i2t / self.contrastive_temp
            loss_alignment = 0.5 * (
                F.cross_entropy(logits_t2i, align_targets.to(device))
                + F.cross_entropy(logits_i2t, align_targets.to(device))
            )
        else:
            text_proj = outputs.get("text_proj")
            image_proj = outputs.get("image_proj")
            orig_mask = batch["binary_labels"] == 0
            if text_proj is not None and image_proj is not None and orig_mask.any():
                orig_indices = orig_mask.nonzero(as_tuple=False).squeeze(-1).to(device=device)
                logits_t2i = text_proj[orig_indices] @ image_proj.transpose(0, 1)
                logits_i2t = image_proj[orig_indices] @ text_proj.transpose(0, 1)
                logits_t2i = logits_t2i / self.contrastive_temp
                logits_i2t = logits_i2t / self.contrastive_temp
                loss_alignment = 0.5 * (
                    F.cross_entropy(logits_t2i, orig_indices)
                    + F.cross_entropy(logits_i2t, orig_indices)
                )

        # Image-text matching over pristine matched pairs versus mismatched pairs
        loss_itm = torch.tensor(0.0, device=device)
        itm_logits = outputs.get("itm_logits")
        itm_labels = outputs.get("itm_labels")
        if itm_logits is None or itm_labels is None:
            itm_pos_logits = outputs.get("itm_pos_logits")
            itm_neg_logits = outputs.get("itm_neg_logits")
            if itm_pos_logits is not None and itm_neg_logits is not None:
                itm_logits = torch.cat([itm_pos_logits, itm_neg_logits], dim=0)
                itm_labels = torch.cat([
                    torch.ones(itm_pos_logits.shape[0], dtype=torch.long, device=device),
                    torch.zeros(itm_neg_logits.shape[0], dtype=torch.long, device=device),
                ], dim=0)
        if itm_logits is not None and itm_labels is not None:
            loss_itm = F.cross_entropy(itm_logits, itm_labels)

        loss_consist = torch.tensor(0.0, device=device)

        total = (
            self.lambda_binary * loss_binary
            + self.lambda_type * loss_type
            + self.lambda_text_grounding * loss_tg
            + self.lambda_image_grounding * loss_ig
            + self.lambda_alignment * loss_alignment
            + self.lambda_itm * loss_itm
        )

        return {
            "total": total,
            "binary": loss_binary.detach(),
            "type": loss_type.detach(),
            "text_grounding": loss_tg.detach() if isinstance(loss_tg, torch.Tensor) else torch.tensor(0.0),
            "image_grounding": loss_ig.detach() if isinstance(loss_ig, torch.Tensor) else torch.tensor(0.0),
            "alignment": loss_alignment.detach(),
            "itm": loss_itm.detach(),
            "consistency": loss_consist.detach(),
        }
