import json
import os

from PIL import Image
from torch.utils.data import Dataset

from src.data.label_utils import (
    get_binary_label,
    get_type_label,
    has_image_manipulation,
    has_text_manipulation,
)
from src.data.transforms import DGM4Transform


class DGM4Dataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        annotation_file: str,
        image_size: int = 224,
        is_train: bool = True,
    ):
        self.dataset_root = dataset_root
        self.transform = DGM4Transform(image_size=image_size, is_train=is_train)

        ann_path = os.path.join(dataset_root, annotation_file)
        with open(ann_path, "r") as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        fake_cls = ann["fake_cls"]

        # Load image
        image_path = self._resolve_image_path(ann["image"])
        image = Image.open(image_path).convert("RGB")

        # Get bbox if present
        bbox = ann.get("fake_image_box")
        if bbox is not None and (not isinstance(bbox, list) or len(bbox) != 4):
            bbox = None

        # Apply transforms (tracks bbox automatically)
        image_tensor, transformed_bbox = self.transform(image, bbox)

        fake_text_pos = ann.get("fake_text_pos", [])
        has_img = has_image_manipulation(fake_cls) and transformed_bbox is not None
        has_txt = has_text_manipulation(fake_cls) and len(fake_text_pos) > 0

        return {
            "image": image_tensor,
            "text": ann["text"],
            "binary_label": get_binary_label(fake_cls),
            "type_label": get_type_label(fake_cls),
            "fake_cls": fake_cls,
            "fake_image_box": transformed_bbox,
            "fake_text_pos": fake_text_pos,
            "has_image_grounding": has_img,
            "has_text_grounding": has_txt,
        }

    def _resolve_image_path(self, json_image_path: str) -> str:
        """Handle DGM4 zip extraction nesting quirk."""
        rel_path = json_image_path.replace("DGM4/", "", 1)
        full_path = os.path.join(self.dataset_root, rel_path)
        if os.path.exists(full_path):
            return full_path

        # Insert extra nesting: manipulation/infoswap/file.jpg ->
        # manipulation/infoswap/infoswap/file.jpg
        parts = rel_path.split("/")
        if len(parts) >= 3:
            parts.insert(2, parts[1])
            nested_path = os.path.join(self.dataset_root, *parts)
            if os.path.exists(nested_path):
                return nested_path

        raise FileNotFoundError(f"Cannot resolve: {json_image_path}")
