import torch
import pytorch_lightning as pl

from src.model.text_encoder import TextEncoder
from src.model.image_encoder import ImageEncoder
from src.model.cross_attention import BidirectionalCrossAttention
from src.model.heads import BinaryHead, TypeHead, TextGroundingHead, ImageGroundingHead
from src.losses.multi_task_loss import MultiTaskLoss


class CMCNet(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters({"config": config})

        model_cfg = config["model"]
        loss_cfg = config["loss"]

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
        self.binary_head = BinaryHead(fused_dim)
        self.type_head = TypeHead(fused_dim, model_cfg["num_classes"])
        self.text_grounding_head = TextGroundingHead(model_cfg["hidden_dim"])
        self.image_grounding_head = ImageGroundingHead(model_cfg["hidden_dim"])

        # Loss
        self.loss_fn = MultiTaskLoss(loss_cfg)

        # Apply freeze strategy
        self._apply_freeze(config["training"])

    def _apply_freeze(self, training_cfg: dict):
        self._gradual_unfreeze = False
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
            self.print(f"[Epoch {epoch}] Unfroze top-{target} encoder layers")

    def on_validation_epoch_start(self):
        # Reset epoch-level metric accumulators
        self._val_tp = 0
        self._val_fp = 0
        self._val_fn = 0
        self._val_type_correct = 0
        self._val_type_total = 0

    def forward(self, batch: dict) -> dict:
        text_feats = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        image_feats = self.image_encoder(batch["images"])

        target_dtype = self.cross_attention.layers[0].norm_t2i_q.weight.dtype
        text_feats = text_feats.to(target_dtype)
        image_feats = image_feats.to(target_dtype)

        text_padding_mask = ~batch["attention_mask"].bool()
        text_fused, image_fused = self.cross_attention(
            text_feats, image_feats, text_padding_mask
        )

        # Pool CLS tokens from both modalities
        fused_cls = torch.cat([
            text_fused[:, 0, :],
            image_fused[:, 0, :],
        ], dim=-1)  # (B, 1536)

        return {
            "binary_logits": self.binary_head(fused_cls),
            "type_logits": self.type_head(fused_cls),
            "text_ground_logits": self.text_grounding_head(text_fused),
            "image_ground_logits": self.image_grounding_head(
                image_fused[:, 1:, :]  # Skip CLS
            ),
        }

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
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
        self.log("train/consistency_loss", losses["consistency"])

        # Log binary accuracy
        preds = outputs["binary_logits"].argmax(dim=-1)
        acc = (preds == batch["binary_labels"]).float().mean()
        self.log("train/binary_acc", acc, prog_bar=True)

        return total

    def on_before_optimizer_step(self, optimizer):
        # Log gradient norm AFTER backward, BEFORE optimizer step
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

        # Per-batch accuracy (averaged by Lightning across epoch)
        preds = outputs["binary_logits"].argmax(dim=-1)
        acc = (preds == batch["binary_labels"]).float().mean()
        self.log("val/binary_acc", acc, prog_bar=True, sync_dist=True)

        # Accumulate counts for epoch-level F1 (NOT per-batch)
        self._val_tp += ((preds == 1) & (batch["binary_labels"] == 1)).sum().item()
        self._val_fp += ((preds == 1) & (batch["binary_labels"] == 0)).sum().item()
        self._val_fn += ((preds == 0) & (batch["binary_labels"] == 1)).sum().item()

        # Type accuracy accumulation
        type_preds = outputs["type_logits"].argmax(dim=-1)
        self._val_type_correct += (type_preds == batch["type_labels"]).sum().item()
        self._val_type_total += batch["type_labels"].shape[0]

        return losses["total"]

    def on_validation_epoch_end(self):
        # Epoch-level binary F1 from accumulated counts
        tp, fp, fn = self._val_tp, self._val_fp, self._val_fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        self.log("val/binary_f1", f1, prog_bar=True)
        self.log("val/binary_precision", precision)
        self.log("val/binary_recall", recall)

        # Epoch-level type accuracy
        type_acc = self._val_type_correct / max(self._val_type_total, 1)
        self.log("val/type_acc", type_acc)

    def configure_optimizers(self):
        config = self.hparams["config"]["training"]

        if config.get("freeze_encoders", False):
            # Phase 1: only trainable params
            trainable = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=config["lr"], weight_decay=0.01)
        else:
            # Phase 2: three param groups with discriminative LR.
            # Include ALL encoder params (even currently frozen ones) so that
            # gradual unfreezing doesn't leave them outside the optimizer.
            encoder_params = []
            fusion_params = []
            head_params = []

            for name, param in self.named_parameters():
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

            optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

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
