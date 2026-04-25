from copy import deepcopy

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

from src.model.text_encoder import TextEncoder
from src.model.image_encoder import ImageEncoder
from src.model.cross_attention import BidirectionalCrossAttention
from src.model.heads import (
    BinaryHead,
    ProjectionHead,
    TypeHead,
    TextGroundingHead,
    ITMHead,
    ImageGroundingHead,
)
from src.losses.multi_task_loss import MultiTaskLoss, bbox_from_logits


def _box_iou_diagonal(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    """IoU for aligned normalized xyxy boxes."""
    lt = torch.maximum(pred_boxes[:, :2], target_boxes[:, :2])
    rb = torch.minimum(pred_boxes[:, 2:], target_boxes[:, 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[:, 0] * wh[:, 1]

    pred_wh = (pred_boxes[:, 2:] - pred_boxes[:, :2]).clamp(min=0.0)
    target_wh = (target_boxes[:, 2:] - target_boxes[:, :2]).clamp(min=0.0)
    pred_area = pred_wh[:, 0] * pred_wh[:, 1]
    target_area = target_wh[:, 0] * target_wh[:, 1]
    union = pred_area + target_area - inter
    return inter / union.clamp(min=1e-6)


class CMCNet(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters({"config": config})

        model_cfg = config["model"]
        loss_cfg = config["loss"]
        self.momentum = model_cfg.get("momentum", 0.995)
        self.queue_size = model_cfg.get("queue_size", 4096)
        self.itm_start_epoch = model_cfg.get("itm_start_epoch", 1)

        # Encoders
        self.text_encoder = TextEncoder(model_cfg["text_encoder_name"])
        self.image_encoder = ImageEncoder(
            model_cfg["image_encoder_name"],
            model_cfg["image_encoder_pretrained"],
        )

        # Fusion
        self.cross_attention = BidirectionalCrossAttention(
            num_layers=model_cfg["num_cross_attn_layers"],
            hidden_dim=model_cfg["hidden_dim"],
            num_heads=model_cfg["num_cross_attn_heads"],
            ffn_dim=model_cfg["cross_attn_ffn_dim"],
            dropout=model_cfg["cross_attn_dropout"],
        )

        # Heads
        fused_dim = model_cfg["hidden_dim"] * 2  # 1536
        projection_dim = model_cfg.get("projection_dim", 256)
        projection_dropout = model_cfg.get("projection_dropout", 0.1)
        self.binary_head = BinaryHead(fused_dim + 1)
        self.text_projection = ProjectionHead(
            model_cfg["hidden_dim"], projection_dim, projection_dropout
        )
        self.image_projection = ProjectionHead(
            model_cfg["hidden_dim"], projection_dim, projection_dropout
        )
        self.type_head = TypeHead(fused_dim, model_cfg["num_classes"])
        self.text_grounding_head = TextGroundingHead(model_cfg["hidden_dim"])
        self.itm_head = ITMHead(fused_dim)
        self.image_grounding_head = ImageGroundingHead(model_cfg["hidden_dim"])
        self.momentum_text_encoder = deepcopy(self.text_encoder)
        self.momentum_image_encoder = deepcopy(self.image_encoder)
        self.momentum_text_projection = deepcopy(self.text_projection)
        self.momentum_image_projection = deepcopy(self.image_projection)
        self._freeze_momentum_modules()

        self.register_buffer("text_queue", torch.zeros(self.queue_size, projection_dim))
        self.register_buffer("image_queue", torch.zeros(self.queue_size, projection_dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_filled", torch.zeros(1, dtype=torch.long))

        # Loss
        self.loss_fn = MultiTaskLoss(loss_cfg)

        # Apply freeze strategy
        self._apply_freeze(config["training"])

    def train(self, mode: bool = True):
        super().train(mode)
        self.momentum_text_encoder.eval()
        self.momentum_image_encoder.eval()
        self.momentum_text_projection.eval()
        self.momentum_image_projection.eval()
        return self

    def _apply_freeze(self, training_cfg: dict):
        self._gradual_unfreeze = False
        # Step counter for per-unfreeze LR ramp. None = no ramp active.
        # Set to 0 each time a new encoder layer transitions to trainable.
        self._steps_since_unfreeze = None
        if training_cfg.get("freeze_encoders", False):
            self.text_encoder.freeze()
            self.image_encoder.freeze()
        else:
            n = training_cfg.get("unfreeze_top_n_layers", 0)
            if n > 0 and training_cfg.get("gradual_unfreeze", True):
                # Gradual: start fully frozen, unfreeze 1 layer per epoch
                self.text_encoder.freeze()
                self.image_encoder.freeze()
                self._gradual_unfreeze = True
                self._unfreeze_total = n
                self._unfrozen_so_far = 0
            elif n > 0:
                # All at once (legacy behavior)
                self.text_encoder.freeze()
                self.image_encoder.freeze()
                self.text_encoder.unfreeze_top_n_layers(n)
                self.image_encoder.unfreeze_top_n_layers(n)

    def _freeze_momentum_modules(self):
        for module in (
            self.momentum_text_encoder,
            self.momentum_image_encoder,
            self.momentum_text_projection,
            self.momentum_image_projection,
        ):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for online_module, momentum_module in (
            (self.text_encoder, self.momentum_text_encoder),
            (self.image_encoder, self.momentum_image_encoder),
            (self.text_projection, self.momentum_text_projection),
            (self.image_projection, self.momentum_image_projection),
        ):
            for param_online, param_momentum in zip(
                online_module.parameters(), momentum_module.parameters()
            ):
                param_momentum.data.mul_(self.momentum).add_(
                    param_online.data, alpha=1.0 - self.momentum
                )

    @torch.no_grad()
    def _dequeue_and_enqueue(
        self,
        text_proj: torch.Tensor | None,
        image_proj: torch.Tensor | None,
    ):
        if text_proj is None or image_proj is None or text_proj.numel() == 0:
            return

        text_proj = text_proj.detach().to(self.text_queue.device, dtype=self.text_queue.dtype)
        image_proj = image_proj.detach().to(self.image_queue.device, dtype=self.image_queue.dtype)

        batch_size = text_proj.shape[0]
        if batch_size > self.queue_size:
            text_proj = text_proj[-self.queue_size:]
            image_proj = image_proj[-self.queue_size:]
            batch_size = self.queue_size

        ptr = int(self.queue_ptr.item())
        end = ptr + batch_size

        if end <= self.queue_size:
            self.text_queue[ptr:end] = text_proj
            self.image_queue[ptr:end] = image_proj
        else:
            first = self.queue_size - ptr
            second = end - self.queue_size
            self.text_queue[ptr:] = text_proj[:first]
            self.image_queue[ptr:] = image_proj[:first]
            self.text_queue[:second] = text_proj[first:]
            self.image_queue[:second] = image_proj[first:]

        self.queue_ptr[0] = end % self.queue_size
        self.queue_filled[0] = min(self.queue_size, int(self.queue_filled.item()) + batch_size)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not isinstance(outputs, torch.Tensor):
            self._queue_text_proj = None
            self._queue_image_proj = None
            return
        queue_text = getattr(self, "_queue_text_proj", None)
        queue_image = getattr(self, "_queue_image_proj", None)
        self._dequeue_and_enqueue(queue_text, queue_image)
        self._queue_text_proj = None
        self._queue_image_proj = None

    def on_train_epoch_start(self):
        if not self._gradual_unfreeze:
            return
        epoch = self.current_epoch
        # Epoch 0: keep encoders frozen (let optimizer warm up on fusion/heads)
        # Epoch 1: unfreeze top-1, Epoch 2: top-2, ...
        target = min(epoch, self._unfreeze_total)
        if target > self._unfrozen_so_far:
            self.text_encoder.freeze()
            self.image_encoder.freeze()
            self.text_encoder.unfreeze_top_n_layers(target)
            self.image_encoder.unfreeze_top_n_layers(target)
            self._unfrozen_so_far = target
            # Reset ramp: freshly-unfrozen layer has m=v=0 Adam moments;
            # gently scale encoder LR from ~0 to target over 200 opt steps.
            self._steps_since_unfreeze = 0
            self.print(f"[Epoch {epoch}] Unfroze top-{target} encoder layers")

    def on_validation_epoch_start(self):
        # Reset epoch-level metric accumulators
        self._val_tp = 0
        self._val_tn = 0
        self._val_fp = 0
        self._val_fn = 0
        self._val_type_correct = 0
        self._val_type_total = 0
        self._val_type_element_correct = 0
        self._val_type_element_total = 0
        self._val_image_iou_sum = 0.0
        self._val_image_iou50_count = 0
        self._val_image_iou_total = 0
        self._val_itm_correct = 0
        self._val_itm_total = 0
        self._val_itm_pos_correct = 0
        self._val_itm_pos_total = 0
        self._val_itm_neg_correct = 0
        self._val_itm_neg_total = 0
        self._val_probs = []
        self._val_labels = []
        self._val_type_preds = []
        self._val_type_labels = []

    def _fuse_features(
        self,
        text_feats: torch.Tensor,
        image_feats: torch.Tensor,
        text_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_fused, image_fused = self.cross_attention(
            text_feats, image_feats, text_padding_mask
        )
        fused_cls = torch.cat([
            text_fused[:, 0, :],
            image_fused[:, 0, :],
        ], dim=-1)
        return text_fused, image_fused, fused_cls

    def _project_globals(
        self,
        text_feats: torch.Tensor,
        image_feats: torch.Tensor,
        text_projection: ProjectionHead | None = None,
        image_projection: ProjectionHead | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_projection = text_projection or self.text_projection
        image_projection = image_projection or self.image_projection
        text_proj = text_projection(text_feats[:, 0, :].float())
        image_proj = image_projection(image_feats[:, 0, :].float())
        return F.normalize(text_proj, dim=-1), F.normalize(image_proj, dim=-1)

    @torch.no_grad()
    def _get_queue_snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        filled = int(self.queue_filled.item())
        if filled == 0:
            empty = self.text_queue[:0]
            return empty, empty
        return self.text_queue[:filled].detach(), self.image_queue[:filled].detach()

    @torch.no_grad()
    def _encode_momentum_orig(
        self,
        batch: dict,
        orig_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_feats = self.momentum_text_encoder(
            batch["input_ids"][orig_indices],
            batch["attention_mask"][orig_indices],
        )
        image_feats = self.momentum_image_encoder(batch["images"][orig_indices])
        return self._project_globals(
            text_feats,
            image_feats,
            self.momentum_text_projection,
            self.momentum_image_projection,
        )

    @torch.no_grad()
    def _build_alignment_logits(
        self,
        text_proj_orig: torch.Tensor,
        image_proj_orig: torch.Tensor,
        momentum_text_proj_orig: torch.Tensor,
        momentum_image_proj_orig: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_queue, image_queue = self._get_queue_snapshot()
        text_bank = torch.cat([
            momentum_text_proj_orig.to(text_proj_orig.device),
            text_queue.to(text_proj_orig.device, dtype=text_proj_orig.dtype),
        ], dim=0)
        image_bank = torch.cat([
            momentum_image_proj_orig.to(image_proj_orig.device),
            image_queue.to(image_proj_orig.device, dtype=image_proj_orig.dtype),
        ], dim=0)
        targets = torch.arange(text_proj_orig.shape[0], device=text_proj_orig.device)
        logits_t2i = text_proj_orig @ image_bank.transpose(0, 1)
        logits_i2t = image_proj_orig @ text_bank.transpose(0, 1)
        return logits_t2i, logits_i2t, targets

    @torch.no_grad()
    def _select_itm_negative_pairs(
        self,
        momentum_text_proj_orig: torch.Tensor,
        momentum_image_proj_orig: torch.Tensor,
        anchor_indices: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        num_orig = anchor_indices.numel()
        if num_orig < 2:
            return None, None

        sim_i2t = momentum_image_proj_orig @ momentum_text_proj_orig.transpose(0, 1)
        sim_t2i = momentum_text_proj_orig @ momentum_image_proj_orig.transpose(0, 1)
        diag = torch.arange(num_orig, device=anchor_indices.device)
        sim_i2t[diag, diag] = float("-inf")
        sim_t2i[diag, diag] = float("-inf")

        neg_text_local = sim_i2t.argmax(dim=-1)
        neg_image_local = sim_t2i.argmax(dim=-1)
        neg_text_scores = sim_i2t[diag, neg_text_local]
        neg_image_scores = sim_t2i[diag, neg_image_local]
        use_text_negative = neg_text_scores >= neg_image_scores

        selected_text_indices = anchor_indices.clone()
        selected_image_indices = anchor_indices.clone()
        selected_text_indices[use_text_negative] = anchor_indices[neg_text_local[use_text_negative]]
        selected_image_indices[~use_text_negative] = anchor_indices[neg_image_local[~use_text_negative]]
        return selected_text_indices, selected_image_indices

    def forward(self, batch: dict) -> dict:
        text_feats = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        image_feats = self.image_encoder(batch["images"])

        target_dtype = self.cross_attention.layers[0].norm_t2i_q.weight.dtype
        text_feats = text_feats.to(target_dtype)
        image_feats = image_feats.to(target_dtype)

        text_padding_mask = ~batch["attention_mask"].bool()
        text_fused, image_fused, fused_cls = self._fuse_features(
            text_feats, image_feats, text_padding_mask
        )
        text_proj, image_proj = self._project_globals(text_feats, image_feats)
        pair_similarity = (text_proj * image_proj).sum(dim=-1, keepdim=True)
        binary_input = torch.cat([
            fused_cls,
            pair_similarity.to(dtype=fused_cls.dtype),
        ], dim=-1)

        alignment_logits_t2i = None
        alignment_logits_i2t = None
        alignment_targets = None
        itm_logits = None
        itm_labels = None
        queue_text_proj = None
        queue_image_proj = None
        if "binary_labels" in batch:
            orig_indices = (batch["binary_labels"] == 0).nonzero(as_tuple=False).squeeze(-1)
            if orig_indices.numel() > 0:
                momentum_text_proj_orig, momentum_image_proj_orig = self._encode_momentum_orig(
                    batch, orig_indices
                )
                queue_text_proj = momentum_text_proj_orig.detach()
                queue_image_proj = momentum_image_proj_orig.detach()

                alignment_logits_t2i, alignment_logits_i2t, alignment_targets = (
                    self._build_alignment_logits(
                        text_proj[orig_indices],
                        image_proj[orig_indices],
                        momentum_text_proj_orig,
                        momentum_image_proj_orig,
                    )
                )

                itm_active = (not self.training) or (self.current_epoch >= self.itm_start_epoch)
                if itm_active:
                    neg_text_indices, neg_image_indices = self._select_itm_negative_pairs(
                        momentum_text_proj_orig,
                        momentum_image_proj_orig,
                        orig_indices,
                    )
                    if neg_text_indices is not None and neg_image_indices is not None:
                        _, _, neg_fused_cls = self._fuse_features(
                            text_feats[neg_text_indices],
                            image_feats[neg_image_indices],
                            text_padding_mask[neg_text_indices],
                        )
                        itm_logits = torch.cat([
                            self.itm_head(fused_cls[orig_indices]),
                            self.itm_head(neg_fused_cls),
                        ], dim=0)
                        itm_labels = torch.cat([
                            torch.ones(orig_indices.shape[0], dtype=torch.long, device=fused_cls.device),
                            torch.zeros(orig_indices.shape[0], dtype=torch.long, device=fused_cls.device),
                        ], dim=0)

        return {
            "binary_logits": self.binary_head(binary_input),
            "type_logits": self.type_head(fused_cls),
            "text_ground_logits": self.text_grounding_head(text_fused),
            "text_proj": text_proj,
            "image_proj": image_proj,
            "alignment_logits_t2i": alignment_logits_t2i,
            "alignment_logits_i2t": alignment_logits_i2t,
            "alignment_targets": alignment_targets,
            "itm_logits": itm_logits,
            "itm_labels": itm_labels,
            "queue_text_proj": queue_text_proj,
            "queue_image_proj": queue_image_proj,
            "image_ground_logits": self.image_grounding_head(image_fused[:, 1:, :]),
        }

    def training_step(self, batch, batch_idx):
        self._momentum_update()
        outputs = self.forward(batch)
        self._queue_text_proj = outputs.get("queue_text_proj")
        self._queue_image_proj = outputs.get("queue_image_proj")
        losses = self.loss_fn(outputs, batch)

        total = losses["total"]

        # Skip batch if loss is NaN/Inf to prevent gradient corruption
        if not torch.isfinite(total):
            self.print(f"[Step {self.global_step}] NaN/Inf loss detected, skipping batch")
            return None

        self.log("train/loss", total, prog_bar=True)
        self.log("train/binary_loss", losses["binary"])
        self.log("train/type_loss", losses["type"])
        self.log("train/text_grounding_loss", losses["text_grounding"])
        self.log("train/image_grounding_loss", losses["image_grounding"])
        self.log("train/alignment_loss", losses["alignment"])
        self.log("train/itm_loss", losses["itm"])
        self.log("train/consistency_loss", losses["consistency"])

        # Log binary accuracy
        preds = outputs["binary_logits"].argmax(dim=-1)
        acc = (preds == batch["binary_labels"]).float().mean()
        self.log("train/binary_acc", acc, prog_bar=True)

        return total

    def on_before_optimizer_step(self, optimizer):
        # NaN/Inf gradient gate: if any param's grad is non-finite, zero all
        # grads so optimizer.step() is a no-op. Prevents weight corruption
        # from a single bad backward pass (e.g. bf16 overflow in DeBERTa
        # disentangled attention). Loss-level NaN is already caught earlier
        # in training_step via the isfinite(total) check.
        has_bad_grad = False
        for p in self.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                has_bad_grad = True
                break

        if has_bad_grad:
            for p in self.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            self.log("train/skipped_step", 1.0)
        else:
            self.log("train/skipped_step", 0.0)

        # Per-unfreeze LR ramp on encoder group. Decouples the shock of
        # fresh Adam moments on just-activated pretrained layers from the
        # cosine schedule's global LR. Only encoder group (index 0) is
        # ramped; fusion and heads run at scheduler's LR.
        if self._steps_since_unfreeze is not None:
            ramp_steps = 200
            if self._steps_since_unfreeze < ramp_steps:
                factor = (self._steps_since_unfreeze + 1) / ramp_steps
                optimizer.param_groups[0]["lr"] *= factor
            self._steps_since_unfreeze += 1

        # Log gradient norm (0 if we zeroed grads — also useful signal)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (p for p in self.parameters() if p.grad is not None), float("inf")
        )
        self.log("train/grad_norm", grad_norm)

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        losses = self.loss_fn(outputs, batch)

        self.log("val/loss", losses["total"], prog_bar=True, sync_dist=True)
        self.log("val/binary_loss", losses["binary"], sync_dist=True)
        self.log("val/type_loss", losses["type"], sync_dist=True)
        self.log("val/text_grounding_loss", losses["text_grounding"], sync_dist=True)
        self.log("val/image_grounding_loss", losses["image_grounding"], sync_dist=True)
        self.log("val/alignment_loss", losses["alignment"], sync_dist=True)
        self.log("val/itm_loss", losses["itm"], sync_dist=True)

        # Per-batch accuracy (averaged by Lightning across epoch)
        preds = outputs["binary_logits"].argmax(dim=-1)
        probs = torch.softmax(outputs["binary_logits"], dim=-1)[:, 1]
        acc = (preds == batch["binary_labels"]).float().mean()
        self.log("val/binary_acc", acc, prog_bar=True, sync_dist=True)

        # Accumulate counts for epoch-level F1 (NOT per-batch)
        self._val_tp += ((preds == 1) & (batch["binary_labels"] == 1)).sum().item()
        self._val_tn += ((preds == 0) & (batch["binary_labels"] == 0)).sum().item()
        self._val_fp += ((preds == 1) & (batch["binary_labels"] == 0)).sum().item()
        self._val_fn += ((preds == 0) & (batch["binary_labels"] == 1)).sum().item()
        self._val_probs.extend(probs.detach().float().cpu().tolist())
        self._val_labels.extend(batch["binary_labels"].detach().cpu().tolist())

        # 4-label manipulation type accumulation
        type_preds = (torch.sigmoid(outputs["type_logits"]) >= 0.5).long()
        type_labels = batch["type_labels"].long()
        self._val_type_correct += (type_preds == type_labels).all(dim=-1).sum().item()
        self._val_type_total += type_labels.shape[0]
        self._val_type_element_correct += (type_preds == type_labels).sum().item()
        self._val_type_element_total += type_labels.numel()
        self._val_type_preds.extend(type_preds.detach().cpu().tolist())
        self._val_type_labels.extend(type_labels.detach().cpu().tolist())

        itm_logits = outputs.get("itm_logits")
        itm_labels = outputs.get("itm_labels")
        if itm_logits is not None and itm_labels is not None:
            itm_preds = itm_logits.argmax(dim=-1)
            self._val_itm_correct += (itm_preds == itm_labels).sum().item()
            self._val_itm_total += itm_labels.numel()
            pos_mask = itm_labels == 1
            neg_mask = itm_labels == 0
            self._val_itm_pos_correct += ((itm_preds == itm_labels) & pos_mask).sum().item()
            self._val_itm_pos_total += pos_mask.sum().item()
            self._val_itm_neg_correct += ((itm_preds == itm_labels) & neg_mask).sum().item()
            self._val_itm_neg_total += neg_mask.sum().item()

        # Bbox grounding accumulation on image-manipulated samples only.
        has_ig = batch["has_image_grounding"].bool()
        if has_ig.any():
            pred_boxes = bbox_from_logits(outputs["image_ground_logits"][has_ig]).detach().float()
            target_boxes = batch["image_grounding_labels"][has_ig].to(
                device=pred_boxes.device,
                dtype=pred_boxes.dtype,
            )
            image_ious = _box_iou_diagonal(pred_boxes, target_boxes)
            self._val_image_iou_sum += image_ious.sum().item()
            self._val_image_iou50_count += (image_ious >= 0.5).sum().item()
            self._val_image_iou_total += image_ious.numel()

        return losses["total"]

    def on_validation_epoch_end(self):
        # Epoch-level binary F1 from accumulated counts
        tp, tn, fp, fn = self._val_tp, self._val_tn, self._val_fp, self._val_fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        real_acc = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fake_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        balanced_acc = 0.5 * (real_acc + fake_acc)
        positive_rate = (tp + fp) / max(tp + tn + fp + fn, 1)

        auroc = 0.0
        if len(set(self._val_labels)) > 1:
            auroc = float(roc_auc_score(self._val_labels, self._val_probs))

        self.log("val/binary_f1", f1, prog_bar=True)
        self.log("val/binary_precision", precision)
        self.log("val/binary_recall", recall)
        self.log("val/binary_real_acc", real_acc)
        self.log("val/binary_fake_acc", fake_acc)
        self.log("val/balanced_acc", balanced_acc, prog_bar=True)
        self.log("val/binary_auroc", auroc, prog_bar=True)
        self.log("val/pred_positive_rate", positive_rate)

        # Epoch-level type accuracy
        type_acc = self._val_type_correct / max(self._val_type_total, 1)
        type_hamming_acc = self._val_type_element_correct / max(self._val_type_element_total, 1)
        self.log("val/type_acc", type_acc)
        self.log("val/type_hamming_acc", type_hamming_acc)
        type_macro_f1 = f1_score(
            self._val_type_labels,
            self._val_type_preds,
            average="macro",
            zero_division=0,
        )
        self.log("val/type_macro_f1", type_macro_f1, prog_bar=True)

        itm_acc = self._val_itm_correct / max(self._val_itm_total, 1)
        itm_pos_acc = self._val_itm_pos_correct / max(self._val_itm_pos_total, 1)
        itm_neg_acc = self._val_itm_neg_correct / max(self._val_itm_neg_total, 1)
        itm_balanced_acc = 0.5 * (itm_pos_acc + itm_neg_acc)
        self.log("val/itm_acc", itm_acc)
        self.log("val/itm_pos_acc", itm_pos_acc)
        self.log("val/itm_neg_acc", itm_neg_acc)
        self.log("val/itm_balanced_acc", itm_balanced_acc)

        image_iou = self._val_image_iou_sum / max(self._val_image_iou_total, 1)
        image_iou50 = self._val_image_iou50_count / max(self._val_image_iou_total, 1)
        self.log("val/image_grounding_iou", image_iou, prog_bar=True)
        self.log("val/image_grounding_iou50", image_iou50)

    def configure_optimizers(self):
        config = self.hparams["config"]["training"]

        if config.get("freeze_encoders", False):
            # Phase 1: only trainable params
            trainable = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable, lr=config["lr"], weight_decay=0.01, eps=1e-6,
            )
        else:
            # Phase 2: three param groups with discriminative LR.
            # Include ALL encoder params (even currently frozen ones) so that
            # gradual unfreezing doesn't leave them outside the optimizer.
            encoder_params = []
            fusion_params = []
            head_params = []

            for name, param in self.named_parameters():
                if name.startswith("momentum_"):
                    continue
                if "text_encoder" in name or "image_encoder" in name:
                    encoder_params.append(param)
                elif "cross_attention" in name:
                    fusion_params.append(param)
                elif "loss_fn" not in name:
                    head_params.append(param)

            param_groups = [
                {"params": encoder_params, "lr": config.get("encoder_lr", 1e-6)},
                {"params": fusion_params, "lr": config.get("fusion_lr", 1e-5)},
                {"params": head_params, "lr": config.get("head_lr", 3e-5)},
            ]

            optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01, eps=1e-6)

        # Cosine schedule with warmup
        from transformers import get_cosine_schedule_with_warmup

        # Estimate total steps
        if self.trainer and self.trainer.estimated_stepping_batches:
            total_steps = self.trainer.estimated_stepping_batches
        else:
            total_steps = config.get("max_epochs", 10) * config.get("steps_per_epoch", 1000)

        warmup_ratio = config.get("warmup_ratio", 0.1)
        warmup_steps = int(warmup_ratio * total_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
