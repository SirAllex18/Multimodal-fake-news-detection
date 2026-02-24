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
        if training_cfg.get("freeze_encoders", False):
            self.text_encoder.freeze()
            self.image_encoder.freeze()
        else:
            n = training_cfg.get("unfreeze_top_n_layers", 0)
            if n > 0:
                self.text_encoder.freeze()
                self.image_encoder.freeze()
                self.text_encoder.unfreeze_top_n_layers(n)
                self.image_encoder.unfreeze_top_n_layers(n)

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

        self.log("train/loss", losses["total"], prog_bar=True)
        self.log("train/binary_loss", losses["binary"])
        self.log("train/type_loss", losses["type"])
        self.log("train/text_grounding_loss", losses["text_grounding"])
        self.log("train/image_grounding_loss", losses["image_grounding"])
        self.log("train/consistency_loss", losses["consistency"])

        # Log binary accuracy
        preds = outputs["binary_logits"].argmax(dim=-1)
        acc = (preds == batch["binary_labels"]).float().mean()
        self.log("train/binary_acc", acc, prog_bar=True)

        return losses["total"]

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        losses = self.loss_fn(outputs, batch)

        self.log("val/loss", losses["total"], prog_bar=True, sync_dist=True)
        self.log("val/binary_loss", losses["binary"], sync_dist=True)
        self.log("val/type_loss", losses["type"], sync_dist=True)
        self.log("val/text_grounding_loss", losses["text_grounding"], sync_dist=True)
        self.log("val/image_grounding_loss", losses["image_grounding"], sync_dist=True)

        preds = outputs["binary_logits"].argmax(dim=-1)
        acc = (preds == batch["binary_labels"]).float().mean()
        self.log("val/binary_acc", acc, prog_bar=True, sync_dist=True)

        # F1 components
        tp = ((preds == 1) & (batch["binary_labels"] == 1)).sum().float()
        fp = ((preds == 1) & (batch["binary_labels"] == 0)).sum().float()
        fn = ((preds == 0) & (batch["binary_labels"] == 1)).sum().float()
        self.log("val/binary_tp", tp, sync_dist=True, reduce_fx="sum")
        self.log("val/binary_fp", fp, sync_dist=True, reduce_fx="sum")
        self.log("val/binary_fn", fn, sync_dist=True, reduce_fx="sum")

        return losses["total"]

    def on_validation_epoch_end(self):
        tp = self.trainer.callback_metrics.get("val/binary_tp", 0)
        fp = self.trainer.callback_metrics.get("val/binary_fp", 0)
        fn = self.trainer.callback_metrics.get("val/binary_fn", 0)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        self.log("val/binary_f1", f1, prog_bar=True)

    def configure_optimizers(self):
        config = self.hparams["config"]["training"]

        if config.get("freeze_encoders", False):
            # Phase 1: only trainable params
            trainable = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=config["lr"], weight_decay=0.01)
        else:
            # Phase 2: three param groups with discriminative LR
            encoder_params = []
            fusion_params = []
            head_params = []

            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                if "text_encoder" in name or "image_encoder" in name:
                    encoder_params.append(param)
                elif "cross_attention" in name:
                    fusion_params.append(param)
                else:
                    head_params.append(param)

            param_groups = []
            if encoder_params:
                param_groups.append({"params": encoder_params, "lr": config.get("encoder_lr", 5e-6)})
            if fusion_params:
                param_groups.append({"params": fusion_params, "lr": config.get("fusion_lr", 3e-5)})
            if head_params:
                param_groups.append({"params": head_params, "lr": config.get("head_lr", 5e-5)})

            optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

        # Cosine schedule with warmup
        from transformers import get_cosine_schedule_with_warmup

        # Estimate total steps
        if self.trainer and self.trainer.estimated_stepping_batches:
            total_steps = self.trainer.estimated_stepping_batches
        else:
            total_steps = config.get("max_epochs", 10) * config.get("steps_per_epoch", 1000)

        warmup_steps = int(0.05 * total_steps)

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
