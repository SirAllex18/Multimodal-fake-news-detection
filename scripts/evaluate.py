import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.data.dgm4_dataset import DGM4Dataset
from src.data.collator import CMCNetCollator
from src.model.cmc_net import CMCNet
from src.metrics.evaluation import (
    binary_metrics,
    type_metrics,
    grounding_metrics,
    per_category_metrics,
    find_optimal_threshold,
)


def collect_predictions(model, loader, device):
    """Run inference on a dataloader and collect all predictions."""
    all_binary_labels = []
    all_binary_probs = []
    all_type_labels = []
    all_type_preds = []
    all_text_ground_labels = []
    all_text_ground_probs = []
    all_image_ground_labels = []
    all_image_ground_probs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(batch)

            # Binary
            binary_probs = torch.softmax(outputs["binary_logits"], dim=-1)[:, 1]
            all_binary_labels.extend(batch["binary_labels"].cpu().tolist())
            all_binary_probs.extend(binary_probs.cpu().tolist())

            # Type
            type_preds = outputs["type_logits"].argmax(dim=-1)
            all_type_labels.extend(batch["type_labels"].cpu().tolist())
            all_type_preds.extend(type_preds.cpu().tolist())

            # Text grounding (manipulated samples only, excluding special tokens)
            has_tg = batch["has_text_grounding"]
            if has_tg.any():
                tg_logits = outputs["text_ground_logits"][has_tg]
                tg_labels = batch["text_grounding_labels"][has_tg]
                for j in range(tg_logits.shape[0]):
                    valid = tg_labels[j] != -100.0
                    if valid.any():
                        all_text_ground_labels.append(tg_labels[j][valid].cpu().numpy())
                        all_text_ground_probs.append(
                            torch.sigmoid(tg_logits[j][valid]).cpu().numpy()
                        )

            # Image grounding
            has_ig = batch["has_image_grounding"]
            if has_ig.any():
                ig_logits = outputs["image_ground_logits"][has_ig]
                ig_labels = batch["image_grounding_labels"][has_ig]
                for j in range(ig_logits.shape[0]):
                    all_image_ground_labels.append(ig_labels[j].cpu().numpy())
                    all_image_ground_probs.append(
                        torch.sigmoid(ig_logits[j]).cpu().numpy()
                    )

            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx + 1} batches")

    return {
        "binary_labels": all_binary_labels,
        "binary_probs": all_binary_probs,
        "type_labels": all_type_labels,
        "type_preds": all_type_preds,
        "text_ground_labels": all_text_ground_labels,
        "text_ground_probs": all_text_ground_probs,
        "image_ground_labels": all_image_ground_labels,
        "image_ground_probs": all_image_ground_probs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    eval_cfg = config.get("evaluation", {})

    checkpoint_path = eval_cfg.get("checkpoint", "checkpoints/phase2/best.ckpt")
    output_dir = eval_cfg.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)

    collator = CMCNetCollator(
        text_encoder_name=config["model"]["text_encoder_name"],
        max_length=data_cfg["max_text_length"],
        patch_grid=config["model"]["patch_grid"],
    )

    loader_kwargs = dict(
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        collate_fn=collator,
        pin_memory=False,
    )

    # Load model
    model = CMCNet.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Run validation set to find optimal threshold
    print("Running validation set for threshold selection...")
    val_dataset = DGM4Dataset(
        dataset_root=data_cfg["dataset_root"],
        annotation_file=data_cfg["val_json"],
        image_size=data_cfg["image_size"],
        is_train=False,
    )
    val_loader = DataLoader(val_dataset, **loader_kwargs)
    val_preds = collect_predictions(model, val_loader, device)

    opt_thresh, opt_val_f1 = find_optimal_threshold(
        val_preds["binary_labels"], val_preds["binary_probs"]
    )
    print(f"Optimal threshold from validation: {opt_thresh:.2f} (val F1={opt_val_f1:.4f})")

    print("\nRunning test set evaluation...")
    test_dataset = DGM4Dataset(
        dataset_root=data_cfg["dataset_root"],
        annotation_file=data_cfg["test_json"],
        image_size=data_cfg["image_size"],
        is_train=False,
    )
    test_loader = DataLoader(test_dataset, **loader_kwargs)
    test_preds = collect_predictions(model, test_loader, device)

    results = {}
    results["binary_at_0.5"] = binary_metrics(
        test_preds["binary_labels"], test_preds["binary_probs"], threshold=0.5
    )
    results["binary_at_opt_thresh"] = binary_metrics(
        test_preds["binary_labels"], test_preds["binary_probs"], threshold=opt_thresh
    )
    results["optimal_threshold"] = {
        "threshold": opt_thresh,
        "val_f1": opt_val_f1,
        "source": "validation_set",
    }
    results["type"] = type_metrics(test_preds["type_labels"], test_preds["type_preds"])
    results["text_grounding"] = grounding_metrics(
        test_preds["text_ground_labels"], test_preds["text_ground_probs"]
    )
    results["image_grounding"] = grounding_metrics(
        test_preds["image_ground_labels"], test_preds["image_ground_probs"]
    )

    fake_cls_list = [test_dataset.annotations[i]["fake_cls"] for i in range(len(test_dataset))]
    fake_cls_list = fake_cls_list[:len(test_preds["binary_labels"])]
    results["per_category"] = per_category_metrics(
        test_preds["binary_labels"], test_preds["binary_probs"], fake_cls_list,
        threshold=opt_thresh,
    )

    output_path = os.path.join(output_dir, "test_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Binary Detection (threshold=0.5)")
    for k, v in results["binary_at_0.5"].items():
        print(f"  {k}: {v:.4f}")

    print(f"Binary Detection (threshold={opt_thresh:.2f}, tuned on val)")
    for k, v in results["binary_at_opt_thresh"].items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Type Classification ===")
    print(f"  accuracy: {results['type']['accuracy']:.4f}")
    print(f"  macro_f1: {results['type']['macro_f1']:.4f}")

    print("\n=== Text Grounding ===")
    for k, v in results["text_grounding"].items():
        print(f"  {k}: {v:.4f}")

    print("\n=== Image Grounding ===")
    for k, v in results["image_grounding"].items():
        print(f"  {k}: {v:.4f}")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
