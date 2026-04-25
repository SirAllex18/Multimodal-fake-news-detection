import torch
from collections import Counter
from typing import List

FAKE_CLS_LIST = [
    "orig",                          # 0
    "face_swap",                     # 1
    "face_attribute",                # 2
    "text_swap",                     # 3
    "text_attribute",                # 4
    "face_swap&text_swap",           # 5
    "face_swap&text_attribute",      # 6
    "face_attribute&text_swap",      # 7
    "face_attribute&text_attribute", # 8
]

FAKE_CLS_TO_IDX = {name: i for i, name in enumerate(FAKE_CLS_LIST)}

MANIPULATION_TYPES = [
    "face_swap",
    "face_attribute",
    "text_swap",
    "text_attribute",
]


def get_binary_label(fake_cls: str) -> int:
    return 0 if fake_cls == "orig" else 1


def get_type_label(fake_cls: str) -> list[float]:
    """Return DGM4's 4-label manipulation target: FS, FA, TS, TA."""
    return [
        float("face_swap" in fake_cls),
        float("face_attribute" in fake_cls),
        float("text_swap" in fake_cls),
        float("text_attribute" in fake_cls),
    ]


def has_image_manipulation(fake_cls: str) -> bool:
    return "face_swap" in fake_cls or "face_attribute" in fake_cls


def has_text_manipulation(fake_cls: str) -> bool:
    return "text_swap" in fake_cls or "text_attribute" in fake_cls


def compute_class_weights(annotations: List[dict]) -> torch.Tensor:
    """Compute inverse-frequency class weights, normalized to sum=num_classes."""
    counts = Counter()
    for ann in annotations:
        counts[ann["fake_cls"]] += 1

    num_classes = len(FAKE_CLS_LIST)
    total = sum(counts.values())
    weights = torch.zeros(num_classes)
    for cls_name, idx in FAKE_CLS_TO_IDX.items():
        count = counts.get(cls_name, 1)
        weights[idx] = total / (num_classes * count)

    # Normalize so weights sum to num_classes
    weights = weights * (num_classes / weights.sum())
    return weights


def compute_type_pos_weights(annotations: List[dict]) -> torch.Tensor:
    """Compute BCE positive weights for the 4-label manipulation target."""
    positives = torch.zeros(len(MANIPULATION_TYPES))
    for ann in annotations:
        positives += torch.tensor(get_type_label(ann["fake_cls"]), dtype=torch.float32)

    total = float(len(annotations))
    negatives = total - positives
    return negatives / positives.clamp(min=1.0)
