"""Sanity checker: loads N samples, verifies paths, shows tokenization."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from src.utils.config import load_config
from src.data.dgm4_dataset import DGM4Dataset
from src.data.collator import CMCNetCollator
from src.data.label_utils import FAKE_CLS_LIST


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]

    split_json = data_cfg[f"{args.split}_json"]
    dataset = DGM4Dataset(
        dataset_root=data_cfg["dataset_root"],
        annotation_file=split_json,
        image_size=data_cfg["image_size"],
        is_train=False,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Split: {args.split}")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["text_encoder_name"], use_fast=True
    )

    collator = CMCNetCollator(
        text_encoder_name=config["model"]["text_encoder_name"],
        max_length=data_cfg["max_text_length"],
        patch_grid=config["model"]["patch_grid"],
    )

    for i in range(min(args.num_samples, len(dataset))):
        print(f"\n{'='*60}")
        print(f"Sample {i}")
        sample = dataset[i]

        print(f"  fake_cls: {sample['fake_cls']}")
        print(f"  binary_label: {sample['binary_label']}")
        print(f"  type_label: {sample['type_label']}")
        print(f"  has_text_grounding: {sample['has_text_grounding']}")
        print(f"  has_image_grounding: {sample['has_image_grounding']}")
        print(f"  image shape: {sample['image'].shape}")
        print(f"  text: {sample['text'][:100]}...")
        print(f"  fake_text_pos: {sample['fake_text_pos']}")
        print(f"  fake_image_box: {sample['fake_image_box']}")

        # Tokenization check
        encoding = tokenizer(
            sample["text"],
            max_length=data_cfg["max_text_length"],
            truncation=True,
            return_offsets_mapping=True,
        )
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        print(f"  num_tokens: {len(tokens)}")
        print(f"  first 10 tokens: {tokens[:10]}")

    # Collate a batch to verify shapes
    print(f"\n{'='*60}")
    print("Collating a batch...")
    samples = [dataset[i] for i in range(min(4, len(dataset)))]
    batch = collator(samples)
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {v.shape} ({v.dtype})")
        else:
            print(f"  {k}: {type(v)}")


if __name__ == "__main__":
    main()
