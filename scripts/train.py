import argparse
import os
import sys

import torch
import pytorch_lightning as pl

torch.set_float32_matmul_precision("high")
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.data.dgm4_dataset import DGM4Dataset
from src.data.collator import CMCNetCollator
from src.data.samplers import create_balanced_sampler
from src.data.label_utils import compute_class_weights
from src.model.cmc_net import CMCNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast_dev_run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    training_cfg = config["training"]

    pl.seed_everything(training_cfg.get("seed", 42))

    train_dataset = DGM4Dataset(
        dataset_root=data_cfg["dataset_root"],
        annotation_file=data_cfg["train_json"],
        image_size=data_cfg["image_size"],
        is_train=True,
    )
    val_dataset = DGM4Dataset(
        dataset_root=data_cfg["dataset_root"],
        annotation_file=data_cfg["val_json"],
        image_size=data_cfg["image_size"],
        is_train=False,
    )

    collator = CMCNetCollator(
        text_encoder_name=config["model"]["text_encoder_name"],
        max_length=data_cfg["max_text_length"],
        patch_grid=config["model"]["patch_grid"],
    )

    sampling_cfg = config.get("sampling", {})
    if sampling_cfg.get("use_balanced_sampler", False):
        train_sampler = create_balanced_sampler(
            train_dataset.annotations,
            group_weights=sampling_cfg.get("group_weights"),
        )
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        sampler=train_sampler,
        shuffle=train_shuffle,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collator,
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collator,
        pin_memory=data_cfg.get("pin_memory", True),
    )

    # Model
    model = CMCNet(config)

    # Set class weights for type loss
    class_weights = compute_class_weights(train_dataset.annotations)
    model.loss_fn.set_type_class_weights(class_weights)

    phase = "phase2" if not training_cfg.get("freeze_encoders", True) else "phase1"

    # Resume from checkpoint if specified.
    # For a new phase (different optimizer groups), load weights only and start fresh.
    # For resuming the same phase, pass ckpt_path to Lightning for full state restore.
    ckpt_path = training_cfg.get("resume_from")
    resume_full_state = training_cfg.get("resume_full_state", phase == "phase1")
    if ckpt_path and os.path.exists(ckpt_path):
        if resume_full_state:
            print(f"Resuming full training state from: {ckpt_path}")
        else:
            print(f"Loading weights only from: {ckpt_path} (fresh optimizer/scheduler)")
            import torch as _torch
            ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["state_dict"], strict=False)
            ckpt_path = None
    else:
        if ckpt_path:
            print(f"Warning: checkpoint not found at {ckpt_path}, training from scratch")
        ckpt_path = None
    checkpoint_dir = f"checkpoints/{phase}"

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best",
            monitor="val/binary_f1",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/binary_f1",
            mode="max",
            patience=5,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    logger = TensorBoardLogger("tb_logs", name=f"cmc_net_{phase}")

    trainer = pl.Trainer(
        max_epochs=training_cfg["max_epochs"],
        precision=training_cfg.get("precision", "bf16-mixed"),
        gradient_clip_val=training_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=training_cfg.get("grad_accum", 1),
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=50,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
