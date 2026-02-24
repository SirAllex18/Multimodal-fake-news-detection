import torch
from torchvision.transforms import v2
from torchvision import tv_tensors
from typing import Optional

# CLIP normalization constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class DGM4Transform:
    """Image transform pipeline with automatic bounding box tracking via v2."""

    def __init__(self, image_size: int = 224, is_train: bool = True):
        self.image_size = image_size

        if is_train:
            self.transform = v2.Compose([
                v2.Resize(256),
                v2.RandomCrop(image_size),
                v2.RandomHorizontalFlip(0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ])
        else:
            self.transform = v2.Compose([
                v2.Resize(256),
                v2.CenterCrop(image_size),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ])

    def __call__(
        self,
        image,  # PIL.Image
        bbox: Optional[list] = None,
    ) -> tuple[torch.Tensor, Optional[list]]:
        W, H = image.size

        image_tv = tv_tensors.Image(v2.functional.pil_to_tensor(image))

        if bbox is not None:
            boxes_tv = tv_tensors.BoundingBoxes(
                [bbox],
                format="XYXY",
                canvas_size=(H, W),
            )

            image_out, boxes_out = self.transform(image_tv, boxes_tv)

            bbox_out = boxes_out[0].tolist()

            # Check for degenerate bbox after crop
            if bbox_out[2] <= bbox_out[0] or bbox_out[3] <= bbox_out[1]:
                return image_out, None

            bbox_out = [
                max(0.0, bbox_out[0]),
                max(0.0, bbox_out[1]),
                min(float(self.image_size), bbox_out[2]),
                min(float(self.image_size), bbox_out[3]),
            ]

            return image_out, bbox_out
        else:
            image_out = self.transform(image_tv)
            return image_out, None
