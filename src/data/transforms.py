import torch
from torchvision.transforms import v2
from typing import Optional

# CLIP normalization constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class DGM4Transform:
    """Image transform pipeline with automatic bounding box tracking via v2."""

    def __init__(self, image_size: int = 224, is_train: bool = True):
        self.image_size = image_size
        self.is_train = is_train

    def __call__(
        self,
        image,  # PIL.Image
        bbox: Optional[list] = None,
    ) -> tuple[torch.Tensor, Optional[list]]:
        W, H = image.size

        image_tensor = v2.functional.pil_to_tensor(image).to(torch.float32) / 255.0
        scale = min(self.image_size / H, self.image_size / W)
        new_h = max(1, round(H * scale))
        new_w = max(1, round(W * scale))
        pad_y = (self.image_size - new_h) // 2
        pad_x = (self.image_size - new_w) // 2

        image_resized = v2.functional.resize(
            image_tensor, [new_h, new_w], antialias=True
        )
        image_out = torch.tensor(CLIP_MEAN, dtype=torch.float32).view(3, 1, 1).expand(
            3, self.image_size, self.image_size
        ).clone()
        image_out[:, pad_y:pad_y + new_h, pad_x:pad_x + new_w] = image_resized

        bbox_out = None
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            bbox_out = [
                x1 * scale + pad_x,
                y1 * scale + pad_y,
                x2 * scale + pad_x,
                y2 * scale + pad_y,
            ]
            # Check for degenerate bbox after preprocessing
            if bbox_out[2] <= bbox_out[0] or bbox_out[3] <= bbox_out[1]:
                return image_out, None
            bbox_out = [
                max(0.0, bbox_out[0]),
                max(0.0, bbox_out[1]),
                min(float(self.image_size), bbox_out[2]),
                min(float(self.image_size), bbox_out[3]),
            ]

        if self.is_train and torch.rand(()) < 0.5:
            image_out = torch.flip(image_out, dims=[2])
            if bbox_out is not None:
                x1, y1, x2, y2 = bbox_out
                bbox_out = [
                    self.image_size - x2,
                    y1,
                    self.image_size - x1,
                    y2,
                ]

        image_out = v2.functional.normalize(image_out, mean=CLIP_MEAN, std=CLIP_STD)
        return image_out, bbox_out
