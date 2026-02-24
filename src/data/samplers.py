from torch.utils.data import WeightedRandomSampler
from typing import List


def get_manipulation_group(fake_cls: str) -> str:
    """Classify a sample into one of four balanced sampling groups."""
    if fake_cls == "orig":
        return "orig"
    elif "&" in fake_cls:
        return "mixed"
    elif "text" in fake_cls:
        return "text_only"
    else:
        return "image_only"


def create_balanced_sampler(
    annotations: List[dict],
    group_weights: dict = None,
) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler that upweights minority manipulation types."""
    if group_weights is None:
        group_weights = {
            "orig": 1.0,
            "image_only": 1.5,
            "text_only": 2.5,
            "mixed": 3.0,
        }

    sample_weights = []
    for sample in annotations:
        group = get_manipulation_group(sample["fake_cls"])
        sample_weights.append(group_weights[group])

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
